if __name__ == "__main__":
    import argparse

    import flwr

    # using argparse to parse the argument from command line
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("-n", help="number of rounds")
    args: argparse.Namespace = parser.parse_args()

    flwr.server.start_server(config=flwr.server.ServerConfig(num_rounds=int(args.n) if args.n is not None else 10))
