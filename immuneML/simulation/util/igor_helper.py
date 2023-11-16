import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def make_skewed_model_files(v_genes: list, j_genes: list, original_model_path: Path, new_model_path: Path) -> Path:
    """

    Args:
        j_genes: a list of j gene patterns to be matched against the original model
        v_genes: a list of v gene patterns to be matched against the original model
        original_model_path: path to the folder of the original igor model files
        new_model_path: where to store updated files

    Returns:
        Path to the folder with new model files

    """

    genes = _import_genes_from_model_params(original_model_path)
    _check_if_all_genes_present(genes, v_genes, j_genes)

    if len(v_genes) != 0:
        v_gene_mask = np.array([any(v_gene in gene for v_gene in v_genes) for gene in genes['v']])
        v_gene_weights = v_gene_mask / v_gene_mask.sum()
    else:
        v_gene_weights = None

    if len(j_genes):
        j_gene_mask = np.array([any(j_gene in gene for j_gene in j_genes) for gene in genes['j']])
        j_gene_weights = j_gene_mask / j_gene_mask.sum()
    else:
        j_gene_weights = None

    _make_new_marginals_file(original_model_path, new_model_path, v_gene_weights, j_gene_weights)
    _copy_remaining_model_files(original_model_path, new_model_path)

    return new_model_path


def _copy_remaining_model_files(original_model_path: Path, new_model_path: Path):
    for filename in ['J_gene_CDR3_anchors.csv', 'V_gene_CDR3_anchors.csv', 'model_params.txt']:
        shutil.copy(original_model_path / filename, new_model_path / filename)


def _make_new_marginals_file(original_model_path: Path, new_model_path, v_gene_weights, j_gene_weights):
    with (original_model_path / 'model_marginals.txt').open('r') as file:
        marginals_text = file.readlines()

    if v_gene_weights is not None:
        marginals_text[3] = _get_new_independent_genes(marginals_text, v_gene_weights, gene_name='v', index=0)
    if j_gene_weights is not None:
        marginals_text = _parse_j_genes(marginals_text, j_gene_weights)

    with (new_model_path / 'model_marginals.txt').open('w') as file:
        file.writelines(marginals_text)


def _parse_j_genes(marginals_text: list, j_gene_weights):
    dim_pattern = re.compile("\$Dim\[(\d+)(\,)*(\d*)\]")

    assert "@j_choice" in marginals_text[4]
    hit = re.match(dim_pattern, marginals_text[5])
    assert hit
    if hit.group(3) and hit.group(3) != "":
        marginals_text[5:] = _update_conditional_j_genes(marginals_text[5:], j_gene_weights)
    else:
        marginals_text[7] = _get_new_independent_genes(marginals_text, j_gene_weights, gene_name='j', index=4)

    return marginals_text


def _update_conditional_j_genes(marginals_trunc_list, j_gene_weights) -> list:
    index = 0

    while "@" not in marginals_trunc_list[index] and index < len(marginals_trunc_list):
        assert f"#[v_choice,{index // 2}]" in marginals_trunc_list[index]
        marginals_trunc_list[index + 1] = _recompute_gene_proba_from_line(marginals_trunc_list[index + 1], j_gene_weights)
        index += 2

    return marginals_trunc_list


def _get_new_independent_genes(marginals_text: list, gene_weights, gene_name: str, index: int):
    assert f"@{gene_name}_choice" in marginals_text[index]
    return _recompute_gene_proba_from_line(marginals_text[index + 3], gene_weights)


def _recompute_gene_proba_from_line(line, gene_weights) -> str:
    original_probabilities = np.array(line.split("%")[1].split(","), dtype=float)
    updated_probabilities = original_probabilities * gene_weights
    updated_probabilities = (updated_probabilities / updated_probabilities.sum()).astype(str)
    return "%" + ",".join(updated_probabilities) + "\n"


def _check_if_all_genes_present(original_model_genes, signal_v_genes: list, signal_j_genes: list):
    for gene_name, genes in {'v': signal_v_genes, 'j': signal_j_genes}.items():
        assert all(any(gene in org_gene for org_gene in original_model_genes[gene_name]) for gene in genes), \
            f"Not all {gene_name.upper()} genes in signals are present in the original generative model."


def _import_genes_from_anchor_files(original_model_path: Path) -> dict:
    genes = {"v": [], "j": []}

    for key in genes.keys():
        df = pd.read_csv(original_model_path / f"{key.upper()}_gene_CDR3_anchors.csv")
        genes[key] = df['gene'].values

    return genes


def _import_genes_from_model_params(original_model_path: Path) -> dict:
    genes = {"v": [], "j": []}

    with (original_model_path / 'model_params.txt').open('r') as file:
        lines = file.readlines()

    index = 0

    for gene_name in ["v", "j"]:
        while f"{gene_name.upper()}_gene;" not in lines[index] and lines[index] != "":
            index += 1
        index += 1
        line = lines[index]
        while line[0] != "#" and line != "":
            gene = line.split("%")[1].split(";")[0]
            gene_index = int(line.split("%")[1].split(";")[-1].replace("\n", ""))
            assert gene_name.upper() in gene, (gene, gene_name + " gene")
            genes[gene_name].append((gene, gene_index))
            index += 1
            line = lines[index]

        genes[gene_name] = sorted(genes[gene_name], key=lambda g: g[1])
        genes[gene_name] = [g[0] for g in genes[gene_name]]

    return genes
