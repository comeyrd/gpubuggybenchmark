import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as scistats
from tqdm import tqdm
import scipy.stats as scistats
from typing import Self

def _exp_name_from_csv_path(csv_path):
  return os.path.splitext(os.path.basename(csv_path))[0]
  
def _check_nonempty(df: pd.DataFrame, key: str):
    if df.empty:
        raise ValueError(f"Filtering on '{key}' produced an empty dataframe.")
      
def hashable(item):
  try:
    hash(item)
  except Exception as e:
    return False
  return True

DEFAULT_R2 = 0.36
DEFAULT_SLOPE = 0.048

class PickleWrapper:
  @staticmethod
  def save(path, data):
    with open(path, "wb") as f:
      pickle.dump(data, f)

  @staticmethod
  def load(path):
    if os.path.exists(path):
      with open(path, "rb") as f:
        return pickle.load(f)
    else:
      raise ValueError(f"Error : {path} does not exist")

  @staticmethod
  def getCachedName(path,attributes):
      file_name = _exp_name_from_csv_path(path)
      path = f"temp/{file_name}-{attributes}.pickle"
      return path
  @staticmethod
  def getCachedComparisonName(*paths, attributes):
    if len(paths) < 2:
        raise ValueError("Need at least 2 paths to compare")
    file_names = [_exp_name_from_csv_path(path) for path in paths]
    file_names_sorted = sorted(file_names)
    names_str = "-".join(file_names_sorted)
    path = f"temp/{names_str}-{attributes}.pickle"
    return path
  def exists(pickle_path):
    return os.path.exists(pickle_path)

# Actions on experiment :
# b -> boostrapping
# e -> entropy
class axisFilter:
  axe:str
  used:bool
  explode:bool
  def __init__(self,axe_:str="",used_:bool=False,explode_:bool=False):
    self.axe = axe_
    self.used = used_
    self.explode = explode_
  def copy(self):
    """Create a deep copy of this axisFilter."""
    return axisFilter(self.axe, self.used, self.explode)
  def __str__(self):
    return f'{{axe : {self.axe}, used : {self.used}, exp : {self.explode}}}'

DEFAULT_INPUTS = ["kernel","version","repetitions","warmups","work_size","flush_l2","blocking","exp"]
class Filter:
  defaults:dict = {}
  on_x:axisFilter = axisFilter()
  on_y:axisFilter = axisFilter()
  on_hue:axisFilter = axisFilter()
  on_subplot:axisFilter = axisFilter()
  subsets:dict = {}
  custom_used = []
  def __str__(self):
    return f"{{defaults:{self.defaults}, on_x:{self.on_x}, on_y:{self.on_y}, on_hue:{self.on_hue}, on_subplot:{self.on_subplot}, subsets:{self.subsets}}}"
  def __init__(self,df):
    self.defaults = {}
    self.subsets = {}
    for def_input in DEFAULT_INPUTS:
      if def_input in df.columns : 
        values_l = df[def_input].unique()
        self.defaults[def_input] = values_l[len(values_l)//2]
  @classmethod
  def empty(cls)->Self:
      """Create an empty Filter without requiring a dataframe."""
      instance = object.__new__(cls)
      instance.defaults = {}
      instance.subsets = {}
      instance.on_x = axisFilter()
      instance.on_y = axisFilter()
      instance.on_hue = axisFilter()
      instance.on_subplot = axisFilter()
      instance.custom_used = []
      return instance

  @classmethod
  def from_best_config(cls,df:pd.DataFrame)->Self:
    empty = cls.empty()
    df = df.sort_values(by=EcdfMetrics.ECDF_COMPARISON_SCORE,ascending=False)
    for def_input in DEFAULT_INPUTS:
      if def_input in df.columns :
        empty.defaults[def_input] = df[def_input].iloc[0]
    return empty
  
  def copy(self):
    """Create a deep copy of this Filter."""
    new_filter = Filter.empty()
    new_filter.defaults = self.defaults.copy()
    new_filter.subsets = self.subsets.copy()
    new_filter.on_x = self.on_x.copy()
    new_filter.on_y = self.on_y.copy()
    new_filter.on_hue = self.on_hue.copy()
    new_filter.on_subplot = self.on_subplot.copy()
    return new_filter
    
  def set_default(self,key,val):
    self.defaults[key] = val

  def set_subsets(self,key,val):
    self.subsets[key] = val
    
  def set_x(self,name:str=None,used:bool=None):
    if name is not None:
      self.on_x.axe = name
    if used is not None:
      self.on_x.used = used

  def set_y(self,name:str=None,used:bool=None,explode:bool=None):
    if name is not None:
      self.on_y.axe = name
    if used is not None:
      self.on_y.used = used
    if explode is not None:
      self.on_y.explode = explode

  def set_hue(self,name:str=None,used:bool=None,explode:bool=None):
    if name is not None:
      self.on_hue.axe = name
    if used is not None:
      self.on_hue.used = used
    if explode is not None:
      self.on_hue.explode = explode
      
  def set_subplot(self,name:str=None,used:bool=None):
    if name is not None:
      self.on_subplot.axe = name
    if used is not None:
      self.on_subplot.used = used
      
  def get_used(self):
    used = set()
    for element in self.custom_used:
      used.add(element)
    if self.on_x.used:
      used.add(self.on_x.axe)
    if self.on_y.used:
      used.add(self.on_y.axe)
    if self.on_hue.used:
      if self.on_hue.axe is not None:
        used.add(self.on_hue.axe)
    if self.on_subplot.used:
      if self.on_subplot.axe is not None:
        used.add(self.on_subplot.axe)
    return list(used)
  
  def get_subtitle(self):
    title = "{"
    used = self.get_used()
    for key, val in self.defaults.items():        
      if key not in used:
        title+=f"{key}:{val} ,"
    title = title[:-1] + '}'
    if title == "}" : 
      title = "{}"
    return title
class Experiment:
  source_csv = None
  inner_df = None
  cache_name = ""
  actions = ""
  
  def __init__(self, source_csv_path, actions="be", bootstrap_confidence=0.9, entropy_batch_size=2, entropy_linear_size=25,cache=True):
    self.source_csv = source_csv_path
    self.cache_name = PickleWrapper.getCachedName(self.source_csv,actions)
    if PickleWrapper.exists(self.cache_name):
      self.inner_df = PickleWrapper.load(self.cache_name)
      self.actions = actions
    else:
      self.loadDataframe()
      if 'b' in actions:
        self.do_bootstrap(bootstrap_confidence)
      if 'e' in actions:
        self.do_entropy(entropy_batch_size,entropy_linear_size)
      if cache:
        self.cache_name = PickleWrapper.getCachedName(self.source_csv,self.actions)
        PickleWrapper.save(self.cache_name,self.inner_df)
  @classmethod
  def from_dataframe(cls,dataframe,actions,cache_name):
    instance = cls.__new__(cls)  # Create instance without calling __init__
    instance.source_csv = ""
    instance.inner_df = dataframe.copy()  # or just dataframe if you don't need a copy
    instance.actions = actions
    instance.cache_name = cache_name
    
    if not PickleWrapper.exists(instance.cache_name):
        PickleWrapper.save(instance.cache_name, instance.inner_df)
    
    return instance
    
  
  def loadDataframe(self):
    self.inner_df = pd.read_csv(self.source_csv)
    self.inner_df["warmup_duration"] = self.inner_df["warmup_duration"].apply(
        lambda x: np.array([float(i) for i in str(x).split('|')]))
    self.inner_df["repetitions_duration"] = self.inner_df["repetitions_duration"].apply(
        lambda x: np.array([float(i) for i in x.split('|')]))
    self.inner_df['warmup_duration'] = self.inner_df['warmup_duration'].apply(
        lambda x: [] if np.array_equal(np.asarray(x), np.array([0.0])) else x)

## Adds the folowing rows to the Experiment dataframe :
# ci_low, ci_high, ci_level, bootstrap_data, std_error
  def do_bootstrap(self, confidence=0.9):
    if "b" not in self.actions:
      self.inner_df['ci_low'] = None
      self.inner_df['ci_high'] = None
      self.inner_df['ci_level'] = confidence
      self.inner_df['bootstrap_data'] = None
      self.inner_df['std_error'] = None
      for idx in tqdm(self.inner_df.index):
          conf, data, std = Bootstraping.standard_deviation(
              self.inner_df.loc[idx]["repetitions_duration"], confidence)
          self.inner_df.at[idx, 'ci_low'] = conf.low
          self.inner_df.at[idx, 'ci_high'] = conf.high
          self.inner_df.at[idx, 'bootstrap_data'] = data
          self.inner_df.at[idx, 'std_error'] = std
      self.actions += "b"

## Adds the folowing rows to the Experiment dataframe :
# slope, r2, entropy, stable_entropy, batch_entropy, batch_linear
  def do_entropy(self, batch_entropy_size=2, batch_linear_size=25):
    if "e" not in self.actions:
      self.inner_df

      self.inner_df['slope'] = None
      self.inner_df['r2'] = None
      self.inner_df['entropy'] = None
      self.inner_df['stable_entropy'] = None
      self.inner_df['batch_entropy'] = batch_entropy_size
      self.inner_df['batch_linear'] = batch_linear_size

      for idx in tqdm(self.inner_df.index):
          entropies, r2s, stables, slopes = Entropy.batched_entropy(
              self.inner_df.loc[idx], batch_entropy_size=batch_entropy_size, batch_linear_size=batch_linear_size)
          self.inner_df.at[idx, 'slope'] = slopes
          self.inner_df.at[idx, 'r2'] = r2s
          self.inner_df.at[idx, 'entropy'] = entropies
          self.inner_df.at[idx, 'stable_entropy'] = stables
          self.inner_df
      self.actions += "e"

  
  def filter(self,exp_filter:Filter):
    temp_df = self.inner_df.copy()
    columns = temp_df.columns
    used = exp_filter.get_used()
    duplicates = []
    for key, val in exp_filter.defaults.items():        
      if key in columns and key not in used:
        if val == "Reference":  
          temp_df = temp_df[temp_df[key].str.contains("Reference",na=False)]
        else:
          temp_df = temp_df[temp_df[key]==val]
        _check_nonempty(temp_df,key)
        duplicates.append(key)
    duplicates_used = []
    for key_sub, val_sub in exp_filter.subsets.items():
      if key_sub in columns :
        if not isinstance(val_sub,list):
            temp_df = temp_df[temp_df[key_sub].str.contains(val_sub)]
            _check_nonempty(temp_df,key)
        else:
          temp_df = temp_df[temp_df[key_sub].isin(val_sub)]
          _check_nonempty(temp_df,key_sub)
    for item in used:
      if item not in columns:
        raise ValueError("column on the axis are not in the dataframe :"+item)
      if hashable(temp_df[item].iloc[0]):
        duplicates_used.append(item)
    duplicates.extend(duplicates_used)
    df_unique = temp_df.drop_duplicates(subset=duplicates,keep="first")
    if exp_filter.on_y.explode and exp_filter.on_y.used:
      df_unique = df_unique.explode(exp_filter.on_y.axe)
    return df_unique.copy()

  def fix_values(self,exp_filter:Filter,columns_to_fix:list):
    temp_df = self.inner_df.copy()
    for column in columns_to_fix:
      if column in self.inner_df.columns:
        temp_df = temp_df[temp_df[column]==exp_filter.defaults[column]]
        _check_nonempty(temp_df,column)
    return temp_df
    
  @classmethod
  def concat_experiments(cls, *experiments: Self) -> Self:
      if len(experiments) < 2:
        raise ValueError("Need at least 2 experiments to compare")
      first_actions = experiments[0].actions
      for exp in experiments[1:]:
        if exp.actions != first_actions:
          raise ValueError(
            f"All experiments must share the same actions. "
            f"Found: {[exp.actions for exp in experiments]}"
            )
      csv_names = [exp.source_csv for exp in experiments]
      cache_name = PickleWrapper.getCachedComparisonName(*csv_names, attributes=experiments[0].actions)
      dfs_to_concat = [
        exp.inner_df.assign(exp=_exp_name_from_csv_path(exp.source_csv))
        for exp in experiments
      ]
      newdf = pd.concat(dfs_to_concat, ignore_index=True)
      return cls.from_dataframe(newdf, experiments[0].actions, cache_name)
  
  
  @classmethod
  def compare_experiments_ecdf(cls,exp1:Self,exp2:Self,default_var=DEFAULT_INPUTS)->Self:
    variables = []
    for def_input in DEFAULT_INPUTS:
      if def_input in exp1.inner_df.columns :
         variables.append(def_input)

    df1 = exp1.inner_df.copy()
    df2 = exp2.inner_df.copy()
    df1["name1"] = _exp_name_from_csv_path(exp1.source_csv)
    df2["name2"] = _exp_name_from_csv_path(exp2.source_csv)
    df1.rename(columns={"repetitions_duration":"repetitions_duration1"},inplace=True)
    df2.rename(columns={"repetitions_duration":"repetitions_duration2"},inplace=True)
    df1_small = df1[variables + ["repetitions_duration1","name1"]]
    df2_small = df2[variables + ["repetitions_duration2","name2"]]
    merged = pd.merge(df1_small,df2_small,on=variables,how="inner")
    merged["ks"] = pd.Series(dtype=float)
    merged["total_area"] = pd.Series(dtype=float)
    merged["signed_area"] = pd.Series(dtype=float)
    merged["cvm"] = pd.Series(dtype=float)
    merged["superiority"] = pd.Series(dtype=float)
    for i, row in merged.iterrows():
        ecdf1 = Ecdf(row["repetitions_duration1"])
        ecdf2 = Ecdf(row["repetitions_duration2"])
        merged.loc[i, "ks"] = EcdfMetrics.k_s_test(ecdf1, ecdf2)
        total, signed = EcdfMetrics.ecdf_area(ecdf1, ecdf2)
        merged.loc[i, "total_area"] = total
        merged.loc[i, "signed_area"] = np.abs(signed)
        merged.loc[i, "cvm"] = EcdfMetrics.cvm_distance(ecdf1, ecdf2)
        merged.loc[i, "superiority"] = EcdfMetrics.prob_superiority(ecdf1, ecdf2)
    merged["repetitions_duration_mean_difference"] = np.abs((merged["repetitions_duration1"].apply(np.mean) - merged["repetitions_duration2"].apply(np.mean))/merged["repetitions_duration1"].apply(np.mean))
    final_df = EcdfMetrics.rank_configurations(merged)
    csv_names = [exp1.source_csv,exp2.source_csv]
    cache_name = PickleWrapper.getCachedComparisonName(*csv_names, attributes="ecdf")
    return cls.from_dataframe(final_df, "ecdf", cache_name)


class Bootstraping:
  @staticmethod
  def standard_deviation(array, confidence=0.9):
    res = scistats.bootstrap((array,), np.std, confidence_level=confidence)
    return res.confidence_interval, res.bootstrap_distribution, res.standard_error

class Entropy:
  @staticmethod
  def binned_entropy(measurements, bin_resolution_ms: float = 0.0005) -> float:
    if len(measurements) == 0:
        return 0.0
    epsilon = bin_resolution_ms * 2.0
    binned_keys = np.round(measurements / epsilon) * epsilon
    unique_bins, counts = np.unique(binned_keys, return_counts=True)
    total_samples = len(measurements)
    probabilities = counts / total_samples
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

  @staticmethod
  def stability_metrics(entropy_values: np.ndarray, max_angle_deg: float = DEFAULT_SLOPE,
                        min_r2: float = DEFAULT_R2, window_size=150):
    if len(entropy_values) < 2:
        return False, float(0), float(0)

    x = np.arange(len(entropy_values))
    y = np.array(entropy_values)
    slope, intercept, r_value, _, _ = scistats.linregress(
        x[-window_size:], y[-window_size:])
    r2 = r_value ** 2

    slope_angle_deg = np.degrees(np.arctan(slope))

    slope_check_passed = abs(slope_angle_deg) <= max_angle_deg
    r2_check_passed = r2 >= min_r2
    stable = bool(slope_check_passed and r2_check_passed)

    return stable, float(slope_angle_deg), float(r2)

  ###
  # batch_entropy_size means every x repetitions, we compute the entropy
  # batch_linear_size means every y entropy sample, we compute the linear stats
  #
  ###
  @staticmethod
  def batched_entropy(row, batch_entropy_size=2, batch_linear_size=20, x="repetitions_duration"):
    nb_repetitions = row.repetitions

    entropies = []
    r2s = []
    stables = []
    slopes = []
    count = 0
    for i in range(batch_entropy_size, nb_repetitions, batch_entropy_size):
        entropies.append(Entropy.binned_entropy(row[x][:i]))
        if count % batch_linear_size == 0:
            stable, slope, r2 = Entropy.stability_metrics(entropies)
            stables.append(stable)
            r2s.append(r2)
            slopes.append(slope)
        count += 1
    return entropies, r2s, stables, slopes


class Ecdf:
    sorted_in:list
    base_array:list
    name:str
    def __init__(self,array,name:str=""):
        self.name = name
        self.base_array = array
        sorted = np.sort(array)
        self.sample_size = len(sorted)
        self.sorted_in, counts = np.unique(sorted, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        self.cdf = cumulative_counts / self.sample_size
        
    def at(self, continuous_val):
        index = np.searchsorted(self.sorted_in, continuous_val, side='right') - 1
        # Ensure index is clipped to [-1, last_valid]
        index = np.clip(index, -1, len(self.cdf) - 1)
        
        out = np.zeros_like(index, dtype=float)
        mask = index >= 0
        out[mask] = self.cdf[index[mask]]
        return out

    def plot(self,color=None):
        plt.step(self.sorted_in,self.cdf,color=color,label=self.name)
        

class EcdfMetrics : 
  METRICS_NAMES = ["ks","total_area","signed_area","cvm","superiority"]
  @staticmethod
  def k_s_test(ecdf1: Ecdf, ecdf2: Ecdf) -> float:
      """
      Gives the max difference between the two ecdf
      """
      x = np.unique(np.concatenate([ecdf1.sorted_in, ecdf2.sorted_in]))
      f1 = ecdf1.at(x)
      f2 = ecdf2.at(x)
      return np.max(np.abs(f1 - f2))
    
  @staticmethod
  def ecdf_area(ecdf1: Ecdf, ecdf2: Ecdf):
    """
    Gives back the area between the two curves, 
    returns total_area,signed_area
    """
    x = np.unique(np.concatenate([ecdf1.sorted_in, ecdf2.sorted_in]))
    f1 = ecdf1.at(x)
    f2 = ecdf2.at(x)
    dx = np.diff(x)
    return np.sum(np.abs(f1[:-1] - f2[:-1]) * dx), np.sum((f1[:-1] - f2[:-1]) * dx)

  @staticmethod
  def cvm_distance(ecdf1: Ecdf, ecdf2: Ecdf):
    """
    Computes the two-sample Cramér–von Mises statistic T
    between two empirical CDFs.

    T = (m*n/(m+n)^2) * sum (F1(x) - F2(x))^2 over all points x
    in the combined sorted sample.
    """
    # All unique points from both samples
    x = np.unique(np.concatenate([ecdf1.sorted_in, ecdf2.sorted_in]))

    # Evaluate both ECDFs on the same grid
    F1 = ecdf1.at(x)
    F2 = ecdf2.at(x)

    m = ecdf1.sample_size
    n = ecdf2.sample_size

    # CvM statistic
    T = (m * n / (m + n)**2) * np.sum((F1 - F2)**2)

    return T

  @staticmethod
  def prob_superiority(ecdf1:Ecdf,ecdf2:Ecdf):
    x = np.unique(np.concatenate([ecdf1.sorted_in, ecdf2.sorted_in]))
    F1 = ecdf1.at(x)
    F2 = ecdf2.at(x)
    f2 = np.diff(np.concatenate([[0], F2]))
    return np.sum(f2 * F1)
  
  ECDF_COMPARISON_SCORE = "Ecdf_Comparison_Score"
  
  @staticmethod
  def rank_configurations(dataframe:pd.DataFrame):
    old_columns = dataframe.columns.to_list()
    METRICS_LOWER_IS_BETTER = ['ks', 'total_area', 'signed_area', 'cvm']
    METRICS_CLOSEST_TO_05 = ['superiority']
    all_metrics = []
    for metric in METRICS_CLOSEST_TO_05:
      dataframe[f"{metric}_distance"] = np.abs(dataframe[metric] - 0.5)
      METRICS_LOWER_IS_BETTER.append(f"{metric}_distance")
    
    for metric in METRICS_LOWER_IS_BETTER:
      min_val = dataframe[metric].min()
      max_val = dataframe[metric].max()
      # Normalize and invert: 1 - (M - Min) / (Max - Min)
      dataframe[f'{metric}_norm'] = 1 - (dataframe[metric] - min_val) / (max_val - min_val)
      all_metrics.append(f'{metric}_norm')
      
    dataframe[EcdfMetrics.ECDF_COMPARISON_SCORE] = dataframe.apply(lambda row: sum(row[col] for col in all_metrics),axis=1)
    return dataframe[old_columns+[EcdfMetrics.ECDF_COMPARISON_SCORE]]
  