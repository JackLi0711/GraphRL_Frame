import time
import Environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from pathlib import Path
from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    parser = ArgumentParser()
     
    parser.add_argument("--model_name", type=str, default="Dec_JapanSection_OriginalReward_JapanLoss_JapanCOF_JapanHyperparameters_Episode5000")
    parser.add_argument("--mode", type=str, default="dec", help="Specify inc if increasing the size from the minimum cross-sections; specify dec if reducing the size from the maximum cross-sections.")

    parser.add_argument("--train", action='store_true', default=True, help="True if implement the training. False if only using the pre-trained machine learning model.")
    parser.add_argument("--use_gpu", action='store_true', default=True, help="Use GPU if True, use CPU if False")

    parser.add_argument("--section_type", type=str, default="Japan", help="Section pool, frame geometry: Japan, Taiwan_OldSections, Taiwan_AdjustedMoreSections")
    parser.add_argument("--code_type", type=str, default="Japan", help="Specific code (SCWB): Japan, Taiwan")
    parser.add_argument("--reward_type", type=str, default="original", help="Reward type: original, volume")
    parser.add_argument("--loss_type", type=str, default="Japan", help="Loss calculation: Japan, Taiwan")
    parser.add_argument("--test_model", type=int, default=1, help="Structural model to test the trained model's performance")

    # model
    parser.add_argument("--N_FEATURE", type=int, default=100, help="hidden_dim")

    # buffer
    parser.add_argument("--CAPACITY", type=int, default=1000, help="buffer_size, Japan: 1000, Taiwan: 3000")
    parser.add_argument("--BATCH_SIZE", type=int, default=32, help="batch_size, Japan: 32, Taiwan: 256")
    parser.add_argument("--memorize_frequency", type=int, default=1, help="add_frequency_experience, Japan: 1, Taiwan: 5")

    # training
    parser.add_argument("--GAMMA", type=float, default=0.99, help="gamma")
    parser.add_argument("--TARGET_UPDATE_FREQ", type=int, default=100, help="synchronize_steps, Japan: 100, Taiwan: 50")
    parser.add_argument("--RECORD_INTERVAL", type=int, default=10, help="test_frequency, Japan: 10, Taiwan: 5")
    parser.add_argument("--optimizer", type=str, default="RMSprop", help="optimizer, Japan: RMSprop, Taiwan: Adam")
    parser.add_argument("--epsilon_schedule", type=str, default="constant", help="epsilon schedule, Japan: constant, Taiwan: power_decay")

    parser.add_argument("--n_episode", type=int, default=5000, help="Number of episodes to train the model, Japan: 5000, Taiwan: 500")

    args = parser.parse_args()
    return args


def main(args):
    result_dir = Path(f"result/{args.model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "info.txt", 'w') as f:
        f.write(args.__str__()+"\n\n")


    # Constant epsilon schedule (Japan)
    def constant_epsilon_schedule(episode_number: int,
							   	  constant_epsilon: float=1e-1) -> float:
        return constant_epsilon
    fixed_epsilon_schedule = lambda n: constant_epsilon_schedule(n, 1e-1)

	# Epsilon decay schedule (Taiwan)
    def power_decay_schedule(episode_number: int,
							 decay_factor: float,
							 minimum_epsilon: float) -> float:
        """Power decay schedule found in other practical applications."""
        return max(decay_factor ** episode_number, minimum_epsilon)
    epsilon_decay_schedule = lambda n: power_decay_schedule(n, 0.99, 1e-2)

    if args.epsilon_schedule == "constant":
        epsilon_schedule = fixed_epsilon_schedule
    elif args.epsilon_schedule == "power_decay":
        epsilon_schedule = epsilon_decay_schedule


    env_kwargs = {
        "use_gpu": args.use_gpu,
        "mode": args.mode,
        "model_name": args.model_name,
        
        "section_type": args.section_type,
        "code_type": args.code_type,
        "reward_type": args.reward_type,
        "loss_type": args.loss_type, 

        "n_feature": args.N_FEATURE,
        "capacity": args.CAPACITY,
        "batch_size": args.BATCH_SIZE,
        "memorize_frequency": args.memorize_frequency,
        "gamma": args.GAMMA,
        "target_update_freq": args.TARGET_UPDATE_FREQ,
        "record_interval": args.RECORD_INTERVAL,
        "optimizer": args.optimizer,
        "epsilon_schedule": epsilon_schedule
    }

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