import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.model import EfficientNet
import os
import pandas as pd

def main(config):
    logger = config.get_logger('test')
    input_dir = '/opt/ml/input/data/eval'
    save_dir = '/opt/ml/project-T4193/ENet_Implement/outputs'

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=32,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = EfficientNet.from_name(config['arch']['type'], num_classes=config['arch']['args']['num_classes'])
    logger.info(model)
    
    info_path = os.path.join(input_dir, 'info.csv')
    info = pd.read_csv(info_path)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict) # 이게 checkpoint에 저장된 파라미터 불러오는 코드네

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    preds = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            data = data.to(device)
            # save sample images, or do something with output here
            output = model(data) # output의 타입이 뭔데?
            pred = torch.argmax(output.detach().cpu(), dim=1).numpy()
            preds.extend(pred)

            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
                # total_metrics[i] += metric(output, target) * batch_size

    info['ans'] = preds
    save_path = os.path.join(save_dir, f'submission.csv')
    info.to_csv(save_path, index=False)
    
    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)
    
    


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='/opt/ml/project-T4193/ENet_Implement/config.json', type=str,
                      help='config file path (default: None)')
    # train 한 모델을 가져올 때 사용하는 옵션
    args.add_argument('-r', '--resume', default=None, type=str, 
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
