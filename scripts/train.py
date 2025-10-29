import argparse
import yaml
import sys
sys.path.append('.')
from src.trainer import LEDTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    trainer = LEDTrainer(config)
    trainer.train(resume_from_checkpoint=args.resume)

if __name__ == '__main__':
    main()
