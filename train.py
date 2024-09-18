from args import parse_train_opt
from EDGE import EDGE
import traceback

def train(opt):
    model = EDGE(opt.feature_type)
    try:
        model.train_loop(opt)
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)
