"""GRPO entry point backed by ms-swift + PEFT."""
from ..common import config


def main():
    from .swift_backend import train
    train(config(), 'grpo')


if __name__ == '__main__':
    main()
