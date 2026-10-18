import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fedtriscope.model import FedTriScope
from fedtriscope.trainer import project_source
from fedtriscope.data import CityWindows
from run import main

class CoreTests(unittest.TestCase):
    def test_projection(self):
        t=torch.tensor([1.,0.]); s=torch.tensor([-2.,3.])
        out,_=project_source(s,t)
        torch.testing.assert_close(out,torch.tensor([0.,3.]))
        torch.testing.assert_close(project_source(s,torch.zeros_like(t))[0],s)

    def test_node_independence_and_midnight(self):
        model=FedTriScope(4,288,dim=2).eval()
        x=torch.randn(2,12,4); start=torch.tensor([280,287])
        with torch.no_grad():
            before=model(x,start)
            changed=x.clone(); changed[:,:,1]+=3
            after=model(changed,start)
        self.assertEqual(tuple(before.shape),(2,4,6))
        torch.testing.assert_close(before[:,0],after[:,0],rtol=0,atol=0)
        self.assertTrue(torch.isfinite(after).all())

    def test_no_gap_crossing(self):
        ticks=np.r_[np.arange(10),np.arange(15,25)]
        ds=CityWindows(np.ones((20,2),dtype=np.float32),0,20,3,2,0,1,ticks)
        self.assertEqual(len(ds),12)
        self.assertEqual(ds.dropped_windows,4)

    def test_train_checkpoint_test(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); rng=np.random.RandomState(5)
            for city in ['metr-la','shenzhen']:
                path=root/'data'/city; path.mkdir(parents=True)
                raw=np.empty((200,4,2),dtype=np.float32)
                raw[:,:,0]=rng.uniform(10,30,(200,4))
                raw[:,:,1]=(np.arange(200)%20/20)[:,None]
                np.save(path/'dataset.npy',raw)
            result=main(['--device','cpu','--data_path',str(root/'data'),
                '--data_list','metr-la','shenzhen','--dim','2','--target_days','1',
                '--target_steps_per_day','40','--num_rounds','1','--local_epochs','2',
                '--target_epochs','1','--batch_size','4','--log_dir',str(root/'train')])
            tested=main(['--mode','test','--device','cpu','--data_path',str(root/'data'),
                '--checkpoint',str(root/'train/final.pth'),'--log_dir',str(root/'test')])
            self.assertEqual(tested,result['final_test'])
            curve=json.loads((root/'train/curve.json').read_text())
            self.assertEqual([p['label'] for p in curve],['initial','round_1','finetune_1'])
            self.assertEqual(set(result), {'final_test', 'split'})
            self.assertEqual(set(result['split']), {'train', 'test', 'normalization', 'normalization_note'})
            self.assertEqual(list((root/'train').glob('*.pth')), [root/'train/final.pth'])
            self.assertTrue(all(set(point)=={'label','target_updates','train_loss'} for point in curve))

if __name__=='__main__':
    unittest.main()
