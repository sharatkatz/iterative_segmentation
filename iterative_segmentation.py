"""Main module."""

import numpy as np
import pandas as pd
import os
from collections import Counter
import traceback
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import List
from explainer import ModelPipeline


class Printing:
    def prints(my_str):
        print()
        print('#' * 80)
        print(my_str)
        print('#' * 80)


class InputRecoding:
    def __init__(self, infile, keep_records=None, sample_=None):
        if keep_records:
            Printing.prints(
                f"Limiting down number of records to {keep_records}")
            in_data = pd.read_csv(infile, nrows=keep_records)
        else:
            in_data = pd.read_csv(infile)
        if sample_:
            frac_ = sample_ / len(in_data)
            Printing.prints(
                f"Sampling a fraction of {round(frac_,2)} of original data")
            in_data = in_data.sample(frac=frac_)
        self.raw_file = in_data.copy()
        self.entity_id = self.raw_file.pop("entity_id")
        self.bu_id = self.raw_file.pop("bu_id")
        self.raw_file['frst_purch_dt'] = pd.to_datetime(
            self.raw_file['frst_purch_dt'])
        self.raw_file['last_purch_dt'] = pd.to_datetime(
            self.raw_file['last_purch_dt'])

    def get_float_cols(self) -> List[str]:
        float_cols = self.raw_file.select_dtypes(include=['float64']).columns
        return float_cols

    def get_categorical_cols(self) -> List[str]:
        categorical_cols = self.raw_file.select_dtypes(
            include=['object']).columns
        return categorical_cols

    def get_date_cols(self) -> List[str]:
        date_cols = self.raw_file.select_dtypes(include=['datetime64']).columns
        return date_cols

    def recode_cols(self) -> pd.DataFrame:
        def _recode_flag(in_text) -> bool:
            in_text = str(in_text)
            if in_text.strip() == "N":
                return 0
            else:
                return 1

        Printing.prints('Recoding flag columns')
        flag_columns = []
        for col in self.raw_file.columns:
            if col.endswith('_flg'):
                flag_columns.append(col)
                self.raw_file[col + "_num"] = self.raw_file.apply(
                    lambda row: _recode_flag(row[col]), axis=1)

        # drop original flag categorical columns
        self.raw_file = self.raw_file.drop(flag_columns, axis=1)

        to_dummy_columns = []
        for col in self.raw_file.columns:
            if col.endswith(('_cd', '_desc')):
                print(f'Recoding col: {col}')
                to_dummy_columns.append(col)

        if len(to_dummy_columns):
            # out_dat = self.raw_file.drop(columns=to_dummy_columns, axis=1)
            out_dat = pd.get_dummies(self.raw_file, columns=to_dummy_columns)

        return out_dat


class ImputeSteps:
    def __init__(self, in_data):
        self.in_data = in_data

    def missing_report(self) -> pd.DataFrame:
        return self.in_data.isna().sum() / len(self.in_data)

    def impute_missings(self, impute_columns=[]) -> pd.DataFrame:
        if impute_columns:
            return self.in_data[impute_columns].fillna(0)
        return self.in_data.fillna(0)


class Segments:
    """[Segment data based on methods called]
    """

    def __init__(self, in_data, standardize_input=True):
        self.in_data = in_data
        if standardize_input:
            self.X = StandardScaler().fit_transform(self.in_data.values)
        else:
            self.X = self.in_data.values

    def elbow_visualizer(self) -> None:
        model = KMeans(init='k-means++', max_iter=1000, tol=1e-4,
                       verbose=0, random_state=42, algorithm='auto')

        _from_range = 2
        _to_range = 50
        _metric_range = ['distortion', 'calinski_harabasz', 'silhouette']
        for _metric in _metric_range:
            plt.figure(figsize=(12, 6))
            visualizer = KElbowVisualizer(
    model, k=(
         _from_range, _to_range), timings=True, locate_elbow=True, metric=_metric)

            # Fit the data to the visualizer
            visualizer.fit(self.X)
            visualizer.show()

    def get_clusters(self, *args, **kwargs) -> List[str]:
        cluster_method = args[0].lower().strip()
        Printing.prints(f"Clustering method used.. {cluster_method}")
        X = args[1].values
        if cluster_method == 'kmeans':
            model = KMeans(**kwargs)
        elif cluster_method == 'agglomerative':
            model = AgglomerativeClustering(**kwargs)
        elif cluster_method == 'fuzzy-c-means':
            model = FCM(**kwargs)
        elif cluster_method == 'dbscan':
            model = DBSCAN(**kwargs)
        elif cluster_method == 'birch':
            model = Birch(**kwargs)
        elif cluster_method == 'affinitypropagation':
            model = Birch(**kwargs)
        elif cluster_method == 'spectralclustering':
            model = Birch(**kwargs)
        elif cluster_method == 'gaussianmixture':
            model = GaussianMixture(**kwargs)
        elif cluster_method == 'optics':
            model = OPTICS(**kwargs)
        model.fit(X)
        if cluster_method in (
            'fuzzy-c-means',
            'birch',
            'affinitypropogation',
            'spectralclustering',
                'gaussianmixture'):
            return model.predict(X)
        else:
            return model.labels_.tolist()

    @staticmethod
    def plot_cluster_members(D) -> None:
        try:
            D = dict(sorted(D.items(), key=lambda item: item[1]))
        except TypeError:
            print("plot cluster members function requires dict like input")
        else:
            plt.bar(range(len(D)), list(D.values()), align='center')
            plt.xticks(range(len(D)), list(D.keys()))
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Members")
            plt.show()
        finally:
            print("plot_cluster_members function called")

    def __incremental_c_means(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        # _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'CMeans_' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_c_means(
                'fuzzy-c-means',
                in_data,
                counter,
                out_data,
                n_clusters=n_mix_clusters - 2)

    def incremental_c_means(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated C Means clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_c_means(
                'fuzzy-c-means', in_dataframe, counter, out_data, n_clusters=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def __incremental_k_means(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'KMeans_' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_k_means(
                'KMeans',
                in_data,
                counter,
                out_data,
                n_clusters=n_mix_clusters - 2)

    def incremental_k_means(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated KMeans clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_k_means(
                'KMeans', in_dataframe, counter, out_data, n_clusters=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def __incremental_birch(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'Birch' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_birch(
                'Birch',
                in_data,
                counter,
                out_data,
                n_clusters=n_mix_clusters - 2)

    def incremental_birch(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated Birch clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_birch(
                'Birch', in_dataframe, counter, out_data, n_clusters=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def __incremental_ap(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'AffPropa' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_ap(
                'AffinityPropagation',
                in_data,
                counter,
                out_data,
                n_clusters=n_mix_clusters - 2)

    def incremental_ap(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated Affinity Propagation clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_ap(
                'AffinityPropagation',
                in_dataframe,
                counter,
                out_data,
                n_clusters=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def __incremental_spectral(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'Spectral' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_spectral(
                'SpectralClustering',
                in_data,
                counter,
                out_data,
                n_clusters=n_mix_clusters - 2)

    def incremental_spectral(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated spectral clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_spectral(
                'SpectralClustering',
                in_dataframe,
                counter,
                out_data,
                n_clusters=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def __incremental_gm(self, *args, **kwargs):
        """[summary]

        Returns:
            [type]: [description]
        """
        # Step-1
        cluster_member = list(self.get_clusters(
            *args, **kwargs
        ))
        in_data = args[1]
        counter = args[2]
        out_data = args[3]
        D = Counter(cluster_member)
        Printing.prints(D)
        # _ = Segments.plot_cluster_members(D)
        in_data["Cluster_Num"] = cluster_member
        # Step-2
        n_subjects = len(in_data)
        to_mix_clusters = [
            key for key, value in D.items() if value < 0.02 * n_subjects
        ]
        Printing.prints("Number of cases passed on..")
        print(sum([value for key, value in D.items() if key in to_mix_clusters]))
        Printing.prints("Number of cases clustered in current step..")
        print(sum([value for key, value in D.items()
              if key not in to_mix_clusters]))
        pass_on_clusters = [
            key for key,
            value in D.items() if key not in to_mix_clusters]
        Printing.prints("Cluster numbers passed on..")
        print(pass_on_clusters)
        n_mix_clusters = len(to_mix_clusters)
        Printing.prints(f"Number of clusters formed in this step")
        print(len([key for key, _ in D.items() if key not in to_mix_clusters]))
        if n_mix_clusters < 6:
            print("Returning output from function now....")
            return out_data
        else:
            print("Record data to pass on")
            pass_on_data = in_data.loc[
                in_data.Cluster_Num.isin(pass_on_clusters), :
            ]
            pass_on_data["A_Cluster_Num"] = 'Gaussian_' + \
                ALPHABETS[counter] + pass_on_data["Cluster_Num"].astype(str)
            out_data = out_data.append(pass_on_data)
            print(
                "Re-clustering mix of clusters which has > 10pct or < 2pct of number of members")
            print(f"Number of clusters to re-cluster: {n_mix_clusters}")
            in_data = in_data.loc[
                in_data.Cluster_Num.isin(to_mix_clusters), :
            ]
            counter += 1
            return self.__incremental_gm(
                'GaussianMixture',
                in_data,
                counter,
                out_data,
                n_components=n_mix_clusters - 2)

    def incremental_gm(
            self,
            in_dataframe: pd.DataFrame,
            init_clusters: int,
            verbose: bool = False):
        """[Repeated Gaussian Mixture clustering]

        Args:
            in_dataframe ([pandas dataframe]): [dataframe having columns to use for
            clustering]
            init_clusters ([int]): [initial value to try as number of clusters]
            verbose (bool, optional): [prints output dataframe snapshot and unique
            number of clusters obtained]. Defaults to False.

        Returns:
            [pandas dataframe]: [dataframe having `Cluster_Name` as added column.
            It will return an empty dataframe if the clustering isn't
            performed well based on the conditions]
        """
        counter = 0
        out_data = pd.DataFrame()
        try:
            output_dat = self.__incremental_gm(
                'GaussianMixture',
                in_dataframe,
                counter,
                out_data,
                n_components=init_clusters)
        except Exception as exp:
            print(traceback.format_exc())
            print(exp.__class__, " occured.")
        else:
            if verbose:
                print(output_dat)
                if "Cluster_Num" in output_dat.columns:
                    print(output_dat.A_Cluster_Num.unique())
        return output_dat

    def silhouette_scores(self) -> None:
        for n_clusters in range_n_clusters:
            # Create a subplot with 1 row and 2 columns
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(12, 5)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(self.X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self.X, cluster_labels)
            print(
                "For n_clusters =",
                n_clusters,
                "The average silhouette_score is : {}".format(
                    round(
                        silhouette_avg,
                        3)))

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                self.X, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sorted(sample_silhouette_values[cluster_labels == i])

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the
                # middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

                ax1.set_title("The silhouette plot for the various clusters.")
                ax1.set_xlabel("The silhouette coefficient values")
                ax1.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the
                # values
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax1.set_yticks([])  # Clear the yaxis labels / ticks
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # 2nd Plot showing the actual clusters formed
                colors = cm.nipy_spectral(
                    cluster_labels.astype(float) / n_clusters)
                ax2.scatter(self.X[:, 0], self.X[:, 1], marker='.',
                            s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')

                # Labeling the clusters
                centers = clusterer.cluster_centers_
                # Draw white circles at cluster centers
                ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                            c="white", alpha=1, s=200, edgecolor='k')

                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                                s=50, edgecolor='k')

                ax2.set_title("The visualization of the clustered data.")
                ax2.set_xlabel("Feature space for the 1st feature")
                ax2.set_ylabel("Feature space for the 2nd feature")

                plt.suptitle(
                    ("Silhouette analysis for KMeans clustering on sample data "
                     "with n_clusters = %d" %
                     n_clusters), fontsize=14, fontweight='bold')
        plt.show()
