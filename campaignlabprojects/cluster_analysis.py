
def get_clusters(names, labels):
    n_clusters = max(labels)
    clusters = {}
    for cluster in range(n_clusters):
        mask = labels == cluster
        similar = list(names[mask])
        clusters[cluster] = similar
    return clusters


def cluster_constituencies_kmeans(n_clusters, constit_demog, features):
    X = constit_demog[features]

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)

    kmeans_clusters = get_clusters(constit_demog.index, kmeans.labels_)
    return kmeans_clusters, kmeans.labels_


def gather_data(clusters, constituency_scores, ge17):
    data = pd.DataFrame()

    for cluster in clusters:
        constituencies = clusters[cluster]
        swings = []
        for constit in constituencies:
            swings += [constituency_scores["difference"].loc[constit]]
        sorted_swings = np.sort(swings)

        for constit, swing in zip(constituencies, swings):
            if len(swings) == 1:
                sigma_from_mean = 0
                pos_in_cluster = 0.5
            else:
                sigma_from_mean = (swing - np.mean(swings)) / np.std(swings)
                pos_in_cluster = np.where(sorted_swings == swing)[0][0] / (
                    len(swings) - 1
                )
            line = pd.DataFrame(
                {
                    "ons_id": constit,
                    "constituency": name_from_onsid(constit, ge17),
                    "cluster": cluster,
                    "cluster_size": len(swings),
                    "mean": np.mean(swings),
                    "std": np.std(swings),
                    "change": swing,
                    "sigma_from_mean": sigma_from_mean,
                    "dist_from_mean": swing - np.mean(swings),
                    "pos_in_cluster": pos_in_cluster,
                },
                index=[0],
            )
            data = data.append(line, ignore_index=True, sort=False)
    return data
