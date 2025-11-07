import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as scistats
from tqdm import tqdm
import scipy.stats as scistats
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
      file_name = os.path.splitext(os.path.basename(path))[0]
      path = f"temp/{file_name}-{attributes}.pickle"
      return path
  def exists(pickle_path):
    return os.path.exists(pickle_path)

# Actions on experiment :
# b -> boostrapping
# e -> entropy

class Filter:
  warmups:int
  repetitions:int
  work_size:int
  flush_l2:int
  blocking:int
  kernel:list
  version:list
  unique:list #In here, you should put a list of str with the columns you want to keep unique
  
  def __init__(self,_unique=None, _warmups=None,_repetitions=None,_work_size=None,_flush_l2=None,_blocking=None,_kernels=None,_versions=None):
    self.warmups = _warmups
    self.repetitions = _repetitions
    self.work_size = _work_size
    self.flush_l2 = _flush_l2
    self.blocking = _blocking
    self.kernels = _kernels
    self.versions = _versions
    self.unique = _unique



class Experiment:
  source_csv = None
  inner_df = None
  cache_name = ""
  actions = ""
  y_name = "repetitions_duration"
  
  def __init__(self, source_csv_path, actions="be", bootstrap_confidence=0.9, entropy_batch_size=2, entropy_linear_size=25,cache=True):
    self.source_csv = source_csv_path
    self.cache_name = PickleWrapper.getCachedName(self.source_csv,actions)
    if PickleWrapper.exists(self.cache_name):
      self.inner_df = PickleWrapper.load(self.cache_name)
    else:
      self.loadDataframe()
      if 'b' in actions:
        self.do_bootstrap(bootstrap_confidence)
      if 'e' in actions:
        self.do_entropy(entropy_batch_size,entropy_linear_size)
      if cache:
        self.cache_name = PickleWrapper.getCachedName(self.source_csv,self.actions)
        PickleWrapper.save(self.cache_name,self.inner_df)

  
  def loadDataframe(self):
    self.inner_df = pd.read_csv(self.source_csv)
    self.inner_df["warmup_duration"] = self.inner_df["warmup_duration"].apply(
        lambda x: np.array([float(i) for i in str(x).split('|')]))
    self.inner_df["repetitions_duration"] = self.inner_df["repetitions_duration"].apply(
        lambda x: np.array([float(i) for i in x.split('|')]))
    self.inner_df['warmup_duration'] = self.inner_df['warmup_duration'].apply(
        lambda x: [] if np.array_equal(np.asarray(x), np.array([0.0])) else x)

## Adds the folowing rows to the Experiment dataframe :
# ci_low, ci_high, ci_level, boostrap_data, std_error
  def do_bootstrap(self, confidence=0.9):
    if "b" not in self.actions:
      self.inner_df['ci_low'] = None
      self.inner_df['ci_high'] = None
      self.inner_df['ci_level'] = confidence
      self.inner_df['boostrap_data'] = None
      self.inner_df['std_error'] = None
      for idx in tqdm(self.inner_df.index):
          conf, data, std = Bootstraping.standard_deviation(
              self.inner_df.loc[idx]["repetitions_duration"], confidence)
          self.inner_df.at[idx, 'ci_low'] = conf.low
          self.inner_df.at[idx, 'ci_high'] = conf.high
          self.inner_df.at[idx, 'boostrap_data'] = data
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

  
  def filter(self,exp_filter:Filter,explode_y:bool=True):
    temp_df = self.inner_df.copy()
    if exp_filter.blocking is not None:
      if "blocking" in temp_df.columns:
        temp_df = temp_df[temp_df["blocking"]==exp_filter.blocking]
    if exp_filter.flush_l2 is not None:
      if "flush_l2" in temp_df.columns:
        temp_df = temp_df[temp_df["flush_l2"]==exp_filter.flush_l2]
    if exp_filter.work_size is not None:
      if "work_size" in temp_df.columns:
        temp_df = temp_df[temp_df["work_size"]==exp_filter.work_size]
    if exp_filter.warmups is not None:
      temp_df = temp_df[temp_df["warmups"]==exp_filter.warmups]
    if exp_filter.repetitions is not None:
      temp_df = temp_df[temp_df["repetitions"]==exp_filter.repetitions]
    if exp_filter.versions is not None:
      if not isinstance(exp_filter.versions,list):
        exp_filter.versions = [exp_filter.versions]
      temp_df = temp_df[temp_df["version"].isin(exp_filter.versions)]
    if exp_filter.kernels is not None:
      if not isinstance(exp_filter.kernels,list):
        exp_filter.kernels = [exp_filter.kernels]
      temp_df = temp_df[temp_df["kernel"].isin(exp_filter.kernels)]
    if exp_filter.unique is not None:
      temp_df = temp_df.drop_duplicates(subset=exp_filter.unique, keep="first")
    if explode_y:
      temp_df = temp_df.explode(self.y_name)
    return temp_df


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
