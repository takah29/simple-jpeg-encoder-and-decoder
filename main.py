import numpy as np


def main():
    with open("earthmap.jpg", "rb") as f:
        data = f.read()

    print(data[:100])


if __name__ == "__main__":
    main()
