import itertools

def comb_feats(df, cat_feats = ('gender', 'age', 'phone_brand', 'device_model'), r=2):
    """Combination features"""
    comb_feats = list(map(list, itertools.combinations(cat_feats, r)))
    for i in tqdm(comb_feats):
        df[':'.join(i)] = train[i].astype(str).apply(lambda x: ''.join(x), 1).astype(int)
    return df