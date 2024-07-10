import time
import Environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
     
    parser.add_argument("--model_name", type=str, default="Dec_JapanSection_OriginalReward_JapanLoss_JapanCOF_TestRecover")  # Dec_JapanSection_Episode5000
    parser.add_argument("--mode", type=str, default="dec", help="Specify inc if increasing the size from the minimum cross-sections; specify dec if reducing the size from the maximum cross-sections.")

    parser.add_argument("--train", action='store_true', default=True, help="True if implement the training. False if only using the pre-trained machine learning model.")
    parser.add_argument("--use_gpu", action='store_true', default=True, help="Use GPU if True, use CPU if False")

    parser.add_argument("--country", type=str, default="Japan", help="Country for section pool, frame geometry, specific code (SCWB): Japan, Taiwan")
    parser.add_argument("--reward_type", type=str, default="original", help="Reward type: original, volume")
    parser.add_argument("--loss_type", type=str, default="Japan", help="loss calculation type: Japan, Taiwan")
    parser.add_argument("--test_model", type=int, default=1, help="Structural model to test the trained model's performance")

    parser.add_argument("--n_episode", type=int, default=5000, help="Number of episodes to train the machine learning model.")

    args = parser.parse_args()
    return args


def main(args):
    result_dir = Path(f"result/{args.model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "info.txt", 'w') as f:
        f.write(args.__str__()+"\n\n")

    env_kwargs = {
        "gpu": args.use_gpu,
        "mode": args.mode,
        "model_name": args.model_name,
        "country": args.country, 
        "reward_type": args.reward_type,
        "loss_type": args.loss_type}
    env = Environment.Environment(**env_kwargs)

    if args.train: 
        t1 = time.time()
        env.Train(args.n_episode)
        t2 = time.time()
        with open(result_dir / "info.txt", 'a') as f:
            f.write(f"trained time: {(t2-t1)/3600:.3f} hr")
    
    env.Test(test_model=args.test_model)




if __name__ == "__main__":
    args = parse_args()
    main(args)
