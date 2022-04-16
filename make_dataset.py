import logging
import torch

from dataset import PolyDataset

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    order        = "4"
    wdir         = "datasets/external"
    symmetry     = "4 2 1"
    config_fname = "datasets/raw/ch4-n2-energies.xyz"
    dataset = PolyDataset(wdir=wdir, config_fname=config_fname, order=order, symmetry=symmetry, set_intermolecular_to_zero=True) #, lr_model=lr_model)

    interim_pk_fname = "datasets/interim/poly_{}_{}.pk".format(symmetry.replace(" ", "_"), order)
    logging.info("saving polynomials (pickled) to {}".format(interim_pk_fname))
    dataset.save_pickle(interim_pk_fname)

    interim_json_fname = "datasets/interim/poly_{}_{}.json".format(symmetry.replace(" ", "_"), order)
    logging.info("saving polynomials (readable) to {}".format(interim_json_fname))
    dataset.save_json(interim_json_fname)

