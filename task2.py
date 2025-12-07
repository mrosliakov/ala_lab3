import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

if __name__ == "__main__":
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    

    df = ratings.pivot(index='userId', columns='movieId', values='rating')
    
    df = df.dropna(thresh=50, axis=0)
    df = df.dropna(thresh=50, axis=1)
    
    R = df.fillna(df.mean().mean()).values
    
    user_means = np.mean(R, axis=1).reshape(-1, 1)
    R_demeaned = R - user_means

    U, sigma, Vt = svds(R_demeaned, k=3)
    
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(U[:, 0], U[:, 1], U[:, 2], alpha=0.5)
    ax1.set_title("users")
    
    ax2 = fig.add_subplot(122, projection='3d')
    V = Vt.T
    ax2.scatter(V[:, 0], V[:, 1], V[:, 2], alpha=0.5, c='orange')
    ax2.set_title("movies")
    
    plt.show()

    k = 20
    U_k, s_k, Vt_k = svds(R_demeaned, k=k)
    S_k = np.diag(s_k)
    
    preds = (U_k @ S_k @ Vt_k) + user_means
    pred_df = pd.DataFrame(preds, index=df.index, columns=df.columns)

    mask = ~np.isnan(ratings.pivot(index='userId', columns='movieId', values='rating').loc[df.index, df.columns].values)
    original_vals = R[mask]
    pred_vals = preds[mask]
    print(f"mae (error): {np.mean(np.abs(original_vals - pred_vals)):.4f}")

    def get_recs(uid):
        if uid not in pred_df.index: return None
        
        watched = ratings[ratings['userId'] == uid]['movieId']
        user_preds = pred_df.loc[uid]
        candidates = user_preds[~user_preds.index.isin(watched)]
        
        top_ids = candidates.sort_values(ascending=False).head(10).index
        return movies[movies['movieId'].isin(top_ids)][['title', 'genres']]

    for uid in df.index[:3]:
        print(f"\nuser {uid} recs:")
        print(get_recs(uid))