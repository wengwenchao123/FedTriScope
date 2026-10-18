"""Logging and unmasked original-scale forecasting metrics."""
import json
from pathlib import Path
import torch
from .logging import get_logger

class Evaluation:
    def __init__(self, server, clients, args):
        self.server, self.clients, self.args = server, clients, args
        self.target = next(client for client in clients if client.data.name == args.target_city)
        self.history = []
        self.target_history = []
        self.log_dir = Path(args.log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(str(self.log_dir), name=f'FedTriScope.{self.log_dir}', debug=False)
        self.logger.propagate = False
        self.logger.info('Experiment log path: %s', self.log_dir)
        self.logger.info('Arguments: %s', vars(args))
        for client in clients:
            data = client.data
            self.logger.info('Loss settings | city=%s | %s', data.name, client.loss_settings)
            self.logger.info('Time index | city=%s | source=%s | corrected_rows=%d | gaps=%d | '
                             'dropped_windows train=%d test=%d', data.name, data.time_source,
                             data.time_slot_corrections, data.time_gap_count,
                             data.train.dropped_windows, data.test.dropped_windows)

    @torch.no_grad()
    def test(self):
        self.target.model.eval()
        absolute = torch.zeros(self.args.horizon, dtype=torch.float64)
        squared = torch.zeros_like(absolute)
        count = 0
        for x, start, y in self.target.data.test_loader:
            error = (self.target.predict(x, start).cpu() - y).double()
            absolute += error.abs().sum(dim=(0, 1))
            squared += error.square().sum(dim=(0, 1))
            count += error.shape[0] * error.shape[1]
        mae = (absolute / count).tolist()
        rmse = (squared / count).sqrt().tolist()
        for step, (m, r) in enumerate(zip(mae, rmse), start=1):
            self.logger.info('Test | step %d/%d | MAE=%.3f | RMSE=%.3f',
                             step, self.args.horizon, m, r)
        self.logger.info('Test | average MAE=%.3f | average RMSE=%.3f',
                         sum(mae) / len(mae), sum(rmse) / len(rmse))
        return {'per_horizon': [{'step': i + 1, 'MAE': m, 'RMSE': r}
                                for i, (m, r) in enumerate(zip(mae, rmse))],
                'average': {'MAE': sum(mae) / len(mae), 'RMSE': sum(rmse) / len(rmse)}}

    def save_result(self, metrics):
        output = Path(self.args.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        result = {'args': vars(self.args), 'metrics': metrics, 'history': self.history,
                  'loss_settings': {c.data.name: c.loss_settings for c in self.clients},
                  'target_history': self.target_history,
                  'cities': {c.data.name: {'nodes': c.data.num_nodes,
                             'steps_per_day': c.data.steps_per_day,
                             'train_windows': len(c.data.train),
                             'train_steps': c.data.train.end,
                             'test_start': c.data.test.begin,
                             'time_source': c.data.time_source,
                             'time_slot_corrections': c.data.time_slot_corrections,
                             'time_gap_count': c.data.time_gap_count,
                             'train_dropped_windows': c.data.train.dropped_windows,
                             'test_dropped_windows': c.data.test.dropped_windows,
                             'scaler_mean': c.data.mean, 'scaler_std': c.data.std}
                             for c in self.clients}}
        (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        self.logger.info('Saved metrics: %s', (output / 'metrics.json').resolve())
        # 本次训练/评估已结束，释放文件句柄，允许同一进程再次运行。
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
