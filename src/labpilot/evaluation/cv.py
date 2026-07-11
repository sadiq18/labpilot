from sklearn.model_selection import KFold, StratifiedKFold


def get_cv_splitter(
    n_splits: int,
    random_seed: int,
    stratified: bool = True,
):
    if stratified:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
