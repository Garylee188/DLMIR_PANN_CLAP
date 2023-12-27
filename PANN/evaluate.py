from sklearn import metrics

from utils.pytorch_utils import forward
import numpy as np
from models import *
from dataset import EmoDataset

class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)
        predicted = np.argmax(clipwise_output, 1)
        # target =  np.argmax(target, 1)
        total = target.shape[0]
        correct = (predicted == target).sum().item()
        # average_precision = metrics.average_precision_score(
        #     target, clipwise_output, average=None)

        # auc = metrics.roc_auc_score(target, clipwise_output, average=None)
        
        # statistics = {'average_precision': average_precision, 'auc': auc}
        # statistics = {'accuracy':100 * correct / total}
        return 100 * correct / total
      
      
if __name__ == "__main__":
    model = Cnn14(sample_rate=22050, window_size=1024, 
        hop_size=256, mel_bins=80, fmin=0, fmax=8000, 
        classes_num=20)
    batch_size = 16
    weights = torch.load('model_3/checkpoints/main/sample_rate=22050,window_size=1024,hop_size=256,mel_bins=80,fmin=0,fmax=8000/data_type=full_train/ResNet54/loss_type=clip_bce/balanced=none/augmentation=none/batch_size=128/300_iterations.pth')
    model.load_state_dict(weights['model'])
    test_dataset = EmoDataset('emotion',"20_test.csv")
    eval_test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=batch_size,
                                                  num_workers=0,
                                                  pin_memory=True)
    evaluator = Evaluator(model=model)
    test_statistics = evaluator.evaluate(eval_test_loader)
    print('Validate test Accuracy: {:.3f}'.format(
                    np.mean(test_statistics)))