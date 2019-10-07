#!/bin/bash
from simifucube.toy_snap.create_particle_distribution_toysnap import generate_snap, generate_outname_from_config
from simifucube.create_ifucube import generate_cube
from configparser import ConfigParser
from argparse import ArgumentParser

def main(cli=None):
    parser = ArgumentParser()
    parser.add_argument(dest='configfile', help="Path to the configuration file")
    args = parser.parse_args(cli)

    configurator = ConfigParser()
    configurator.read(args.configfile)
    print(configurator)
    config_toy = configurator['toysnap']
    config_ifu = configurator['IFU']

    config_ifu['snap_name'] = generate_outname_from_config(config_toy)
    generate_snap(config_toy)
    generate_cube(config_ifu)

if __name__ == '__main__':
    main()