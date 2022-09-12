from pprint import pprint

import timm


def main() -> None:
    pprint(timm.list_models('*vit*'))


if __name__ == '__main__':
    main()
