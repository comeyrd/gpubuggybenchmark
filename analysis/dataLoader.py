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
      file_name = os.path.splitext(os.path.basename(path))[0]
      path = f"temp/{file_name}-{attributes}.pickle"
      return path
  @staticmethod
  def getCachedComparisonName(path1,path2,attributes):
      file_name1 = os.path.splitext(os.path.basename(path1))[0]
      file_name2 = os.path.splitext(os.path.basename(path2))[0]
      names = sorted([file_name1,file_name2])
      path = f"temp/{names[0]}-{names[1]}-{attributes}.pickle"
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

DEFAULT_INPUTS = ["repetitions","warmups","work_size","flush_l2","blocking","kernel","version","exp"]
class Filter:
  defaults:dict
  on_x:axisFilter = axisFilter()
  on_y:axisFilter = axisFilter()
  on_hue:axisFilter = axisFilter()
  on_subplot:axisFilter = axisFilter()
  subsets:dict
  def __str__(self):
    return f"{{defaults:{vars(self.defaults)}, on_x:{self.on_x}, on_y:{self.on_y}, on_hue:{self.on_hue}, on_subplot:{self.on_subplot}, subsets:{vars(self.subsets)}}}"
  def __init__(self,df):
    self.defaults = {}
    self.subsets = {}
    for def_input in DEFAULT_INPUTS:
      if def_input in df.columns : 
        values_l = df[def_input].unique()
        self.defaults[def_input] = values_l[len(values_l)//2]
  @classmethod
  def empty(cls):
      """Create an empty Filter without requiring a dataframe."""
      instance = object.__new__(cls)
      instance.defaults = {}
      instance.subsets = {}
      instance.on_x = axisFilter()
      instance.on_y = axisFilter()
      instance.on_hue = axisFilter()
      instance.on_subplot = axisFilter()
      return instance
  
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
    used = []
    if self.on_x.used:
      used.append(self.on_x.axe)
    if self.on_y.used:
      used.append(self.on_y.axe)
    if self.on_hue.used:
      if self.on_hue.axe is not None:
        used.append(self.on_hue.axe)
    if self.on_subplot.used:
      if self.on_subplot.axe is not None:
        used.append(self.on_subplot.axe)
    return used
  
  def get_subtitle(self):
    title = "{"
    used = self.get_used()
    for key, val in self.defaults.items():        
      if key not in used:
        title+=f"{key}:{val} ,"
    title = title[:-1] + '}'    
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
        if isinstance(val,list):
          temp_df = temp_df[temp_df[key].isin(val)]
        else:
          temp_df = temp_df[temp_df[key]==val]
        _check_nonempty(temp_df,key)
        duplicates.append(key)
    duplicates_used = []
    for key_sub, val_sub in exp_filter.subsets.items():
      if key_sub in columns :
        if not isinstance(val_sub,list):
          raise ValueError("Subset that is not a list")
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
    return df_unique

  @classmethod
  def compare_experiments(cls,exp1:Self,exp2:Self)->Self:
    if set(exp1.actions) != set(exp2.actions):
      raise ValueError(f"The experiments do not share the same actions : {exp1.actions} - {exp2.actions}")
    cache_name = PickleWrapper.getCachedComparisonName(exp1.source_csv,exp2.source_csv,exp1.actions)
    newdf = pd.concat([
        exp1.inner_df.assign(exp=os.path.splitext(os.path.basename(exp1.source_csv))[0]),
        exp2.inner_df.assign(exp=os.path.splitext(os.path.basename(exp2.source_csv))[0])
    ], ignore_index=True)
    return cls.from_dataframe(newdf,exp1.actions,cache_name)

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
