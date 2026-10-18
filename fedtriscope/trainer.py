"""Target-guided synchronous gradient training; private embeddings stay local.

TSJT equations (8)-(10), restricted to the shared backbone. Sources are averaged
before projection. Projection is per named parameter tensor (including biases).
This is an experimental adaptation, not a reproduction of the full TSJT model.
"""

import json
import torch
from torch.utils.data import DataLoader, default_collate

from .evaluation import Evaluation


def project_source(source, target):
    """Remove only a source component opposing a nonzero target gradient."""
    if source.shape != target.shape:
        raise ValueError('Gradient shapes differ')
    if not torch.isfinite(source).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite gradient')
    s, t = source.double(), target.double()
    dot, norm = (s * t).sum(), t.square().sum()
    conflict = bool(dot < 0)
    if conflict and norm > 0:
        filtered = (s - dot / norm * t).to(source.dtype)
    else:
        filtered = source.clone()
    return filtered, {
        'conflict': conflict,
        'target_norm': float(norm.sqrt()),
        'source_norm': float(s.norm()),
        'removed_norm': float((s - filtered.double()).norm()),
        'dot_before': float(dot),
        'dot_after': float((filtered.double() * t).sum()),
    }


def local_gradients(client, batch):
    """Backward only: neither shared nor private parameters are stepped here."""
    client.model.train()
    client.model.zero_grad(set_to_none=True)
    x, start, y = batch
    loss = client.criterion(client.predict(x, start), y.to(client.device))
    if not torch.isfinite(loss) or loss.item() > 10000:
        raise FloatingPointError(f'{client.data.name}: invalid loss {loss.item()}')
    loss.backward()
    for name, parameter in client.model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'{client.data.name}: invalid gradient {name}')
    return loss.item(), {name: p.grad.detach().clone()
                         for name, p in client.model.shared_model.named_parameters()}


class FedTSBTrainer(Evaluation):
    def _evaluate(self, loader):
        old_loader = self.target.data.test_loader
        modes = [(module, module.training) for module in self.target.model.modules()]
        old_info = self.logger.info
        try:
            self.target.data.test_loader = loader
            self.logger.info = lambda *a, **kw: None
            # Evaluation must not consume training RNG (including dropout RNG).
            with torch.random.fork_rng():
                torch.manual_seed(92000 + self.args.seed)
                return self.test()
        finally:
            self.logger.info = old_info
            self.target.data.test_loader = old_loader
            for module, training in modes:
                module.training = training

    def _checkpoint(self):
        return {
            'args': vars(self.args),
            'server': self.server.get_weights(),
            'clients': {c.data.name: c.model.state_dict() for c in self.clients},
            'scalers': {c.data.name: {'mean': c.data.mean, 'std': c.data.std}
                        for c in self.clients},
            'time_sources': {c.data.name: c.data.time_source for c in self.clients},
        }

    def train_fl(self):
        args, target = self.args, self.target
        private_order = 'joint'
        sources = [c for c in self.clients if c is not target]
        if not sources:
            raise ValueError('FedTSB requires at least one source city')
        data = target.data
        test_loader = DataLoader(data.test, batch_size=args.batch_size, shuffle=False)
        split = {'train': [0, data.train.end],
                 'test': [data.test.begin, data.test.end], 'normalization': args.normalization,
                 'normalization_note': 'full uses the entire series; train fits training only'}
        (self.log_dir / 'split.json').write_text(json.dumps(split, indent=2))
        private_optimizers = {}
        for c in self.clients:
            private = [p for name, p in c.model.named_parameters()
                       if not name.startswith('shared_model.')]
            private_optimizers[c.data.name] = torch.optim.Adam(private, lr=args.lr_init)
        protocol = dict(source_aggregation='equal city mean, then projection',
                        projection_unit='each named shared parameter tensor',
                        shared_optimizer='direct gradient step; no Adam or momentum',
                        private_optimizer='local Adam', zero_target_gradient='retain source gradient',
                        sync_steps=args.num_rounds * args.local_epochs,
                        target_finetune_epochs=args.target_epochs,
                        evaluation='final model on test split after all training',
                        architecture='no node prompting module', split=split,
                        private_order=private_order,
                        backward_passes_per_sync_per_city=1 if private_order == 'joint' else 2,
                        private_steps_per_sync_per_city=1,
                        adaptation='joint gradients at the current shared and private parameters')
        (self.log_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2))
        curve, steps = [], 0

        def observe(label, loss=None):
            curve.append(dict(label=label, target_updates=steps, train_loss=loss))
            (self.log_dir / 'curve.json').write_text(json.dumps(curve, indent=2))
            self.logger.info('%s | train=%s', label, loss)

        # All clients begin at the same shared point, but keep their own embeddings.
        for c in self.clients:
            c.set_weights(self.server.get_weights(), 'overwrite')
        trace = (self.log_dir / 'gradient_stats.jsonl').open('w', buffering=1)
        try:
            observe('initial')
            for r in range(args.num_rounds):
                losses = {c.data.name: 0.0 for c in self.clients}
                generators = {c.data.name: torch.Generator().manual_seed(
                    args.seed * 1000000 + (61000 if c is target else 71000 + 1000 * i) + r)
                    for i, c in enumerate(self.clients)}
                for b in range(args.local_epochs):
                    gradients = {}
                    for i, c in enumerate(self.clients):
                        count = args.batch_size
                        indices = torch.randperm(len(c.data.train), generator=generators[c.data.name])[:count]
                        batch = default_collate([c.data.train[int(j)] for j in indices[:args.batch_size]])
                        with torch.random.fork_rng():
                            torch.manual_seed(args.seed * 1000000 + 81000 +
                                              (r * args.local_epochs + b) * len(self.clients) + i)
                            loss, gradients[c.data.name] = local_gradients(c, batch)
                        losses[c.data.name] += loss / args.local_epochs
                    target_grad = gradients[target.data.name]
                    updates, stats = {}, {}
                    for name, parameter in self.server.model.named_parameters():
                        source = torch.stack([gradients[c.data.name][name] for c in sources]).mean(0)
                        filtered, diagnostic = project_source(source, target_grad[name])
                        if args.tsb_projection == 'none':
                            filtered = source
                        value = parameter.detach() - args.tsb_target_lr * target_grad[name] - args.tsb_source_lr * filtered
                        if not torch.isfinite(value).all():
                            raise FloatingPointError(f'Nonfinite server update: {name}')
                        updates[name], stats[name] = value, diagnostic
                    # Commit only after all gradients and server updates passed checks.
                    for c in self.clients:
                        private_optimizers[c.data.name].step()
                    self.server.model.load_state_dict(updates)
                    for c in self.clients:
                        c.set_weights(updates, 'overwrite')
                    steps += 1
                    trace.write(json.dumps(dict(round=r+1, batch=b+1, sync_step=steps,
                                               projection=args.tsb_projection, parameters=stats)) + '\n')
                self.history.append(dict(round=r+1, loss=losses))
                observe(f'round_{r+1}', losses[target.data.name])
            # Target adaptation stage (50 epochs in the released presets).
            # Shared Adam has no prior state; retain private Adam moments.
            shared_optimizer = torch.optim.Adam(target.model.shared_model.parameters(), lr=args.lr_init)
            for epoch in range(args.target_epochs):
                generator = torch.Generator().manual_seed(101000 + args.seed * 1000000 + epoch)
                order = torch.randperm(len(data.train), generator=generator).tolist()
                epoch_losses = []
                for offset in range(0, len(order), args.batch_size):
                    batch = default_collate([data.train[j] for j in order[offset:offset+args.batch_size]])
                    loss, _ = local_gradients(target, batch)
                    shared_optimizer.step()
                    private_optimizers[target.data.name].step()
                    steps += 1
                    epoch_losses.append(loss)
                loss = sum(epoch_losses) / len(epoch_losses)
                self.target_history.append(dict(epoch=epoch+1, loss=loss))
                observe(f'finetune_{epoch+1}', loss)
            final_metrics = self._evaluate(test_loader)
            torch.save(self._checkpoint(), self.log_dir / 'final.pth')
            result = dict(final_test=final_metrics, split=split)
            self.logger.info('Final test | %s', final_metrics)
            (self.log_dir / 'results.json').write_text(json.dumps(result, indent=2))
            self.save_result(result)
            return result
        finally:
            trace.close()
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
