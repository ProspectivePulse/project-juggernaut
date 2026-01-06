import argparse
import sys
import os

# Ensure the src directory is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import Config

def main():
    parser = argparse.ArgumentParser(description="Project Juggernaut Controller")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 1. TRAIN COMMAND ---
    train_parser = subparsers.add_parser("train", help="Start training the agent (Laptop)")
    train_parser.add_argument("--timesteps", type=int, default=10000, help="Total training steps")
    train_parser.add_argument("--save_path", help="Path to save model")
    train_parser.add_argument("--load_model", help="resume from this model path")
    train_parser.add_argument("--pi_ip", help="Override Robot IP")

    # --- 2. INFERENCE COMMAND ---
    infer_parser = subparsers.add_parser("inference", help="Run inference with trained model (Laptop)")
    infer_parser.add_argument("--model", help="Path to model file")
    infer_parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    infer_parser.add_argument("--step_delay", type=float, default=0.0)
    infer_parser.add_argument("--pi_ip", help="Override Robot IP")

    # --- 3. SERVER COMMAND ---
    server_parser = subparsers.add_parser("server", help="Start the Robot Server (Raspberry Pi")
    server_parser.add_argument("--port", type=int, help="Override Port")

    # --- 4. CHECK COMMAND ---
    subparsers.add_parser("check", help="Validate Configuration")

    # Parse
    args = parser.parse_args()

    # Route
    if args.command == "train":
        print("Mode: TRAIN")
        import src.client.Train_DQN as trainer
        trainer.train_agent(
            ip=args.pi_ip,
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            load_path=args.load_model
        )
    elif args.command == "inference":
        print("Mode: INFERENCE")
        import src.client.run_inference_client_DQN as infer
        infer.run_inference(
            ip=args.pi_ip,
            model_path=args.model,
            deterministic=args.deterministic,
            step_delay=args.step_delay
        )
    elif args.command == "server":
        print("Mode: SERVER")
        import src.server.pi_server as server
        server.start_server(port=args.port)

    elif args.command == "check":
        Config.validate()

if __name__ == "__main__":
    main()