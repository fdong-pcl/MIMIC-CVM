import pandas as pd
import numpy as np
import os
import logging
import argparse
import re
import glob
from tqdm import tqdm 

# 启用 tqdm 与 Pandas 的集成
tqdm.pandas() 

# ================= 配置区域 (Configuration) =================

# 基础数据根目录 (请根据您的实际路径修改)
BASE_DATA_DIR = '/Users/YOURNAME/Desktop/pcl/Data/CVD_MMData/' 
# 统一的输出根目录
OUTPUT_ROOT = os.path.join(BASE_DATA_DIR, 'output/') 

PATHS = {
    # MIMIC-IV 基础数据路径
    'hosp': os.path.join(BASE_DATA_DIR, 'mimiciv/3.1/hosp'),
    'icu': os.path.join(BASE_DATA_DIR, 'mimiciv/3.1/icu'),
    'note_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/note'),
    'cxr_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/cxr'),
    'ecg_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/ecg'),
    'echo_dir': os.path.join(BASE_DATA_DIR, 'mimiciv/echo'),
    # CXR 报告根目录 (用于读取 .txt 文件)
    'cxr_reports_root': os.path.join(BASE_DATA_DIR, 'mimiciv/cxr'), 
    
    # CVD 类别文件路径 (假设这些文件位于此路径下)
    'CVD_CATEGORY_PATH': '/Users/YOURNAME/Desktop/pcl/Code/MIMICIV_CVD_MMData/mmdata_pipeline/cvd_category/', 

    # --- 最终统一输出路径配置 (使用新的 OUTPUT_ROOT) ---
    'step0_output_dir': os.path.join(OUTPUT_ROOT, 'step0_death_admissionlabel/'),
    'step1_output_dir': os.path.join(OUTPUT_ROOT, 'step1_cvd_filter/'), 
    'step2_output_dir': os.path.join(OUTPUT_ROOT, 'step2_multimodal_matching/'),
}

# Note 阶段文件配置 (基于用户提供的文件名)
NOTE_FILES = {
    'radiology': 'radiology.csv.gz',
    'discharge': 'discharge.csv.gz'
}

# CXR 阶段文件配置 (基于用户提供的文件名)
CXR_FILES = {
    'metadata': 'mimic-cxr-2.0.0-metadata.csv.gz',
    'chexpert': 'mimic-cxr-2.0.0-chexpert.csv.gz',
    'study_list': 'cxr-study-list.csv.gz',
    'record_list': 'cxr-record-list.csv.gz'
}

CXR_CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 
    'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion', 
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# ECG 阶段文件配置 (基于用户提供的文件名)
ECG_FILES = { 
    'machine_measurements': 'machine_measurements.csv',  
    'record_list': 'record_list.csv',                
    'waveform_note_links': 'waveform_note_links.csv'     
}

# Echo 阶段文件配置 (基于用户提供的文件名)
ECHO_FILES = { 
    'record_list': 'echo-record-list.csv',         
    'study_list': 'echo-study-list.csv'          
}

# ================= Log 配置 (Log Setup) =================

def setup_logger(output_dir, log_filename, module_name):
    """设置日志记录器，确保 handlers 被清除以防重复"""
    LOG_DIR = os.path.join(output_dir, 'log')
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    log_filepath = os.path.join(LOG_DIR, log_filename)
    
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # 清除旧 handler 防止重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_filepath, mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ================= ICD 范围处理函数 (ICD Range Utilities) =================
# ... (parse_single_icd_range, is_icd_in_range, build_match_map, read_report_text 保持不变) ...
def parse_single_icd_range(range_str):
    """解析单个 ICD 范围字符串 (e.g., "410–414" 或 "I5A") 为 min, max"""
    if not isinstance(range_str, str) or not range_str:
        return None, None
        
    range_str = range_str.strip()
    
    if '–' in range_str: # 使用 en-dash
        parts = range_str.split('–')
    elif '-' in range_str: # 使用 hyphen
        parts = range_str.split('-')
    else:
        # 单个代码 (e.g., 410, I5A)
        return range_str, range_str

    min_code = parts[0].strip()
    max_code = parts[-1].strip()
    
    return min_code, max_code

def is_icd_in_range(icd_code, icd_version, range_min, range_max):
    """判断单个 ICD 代码是否落在 min/max 范围内"""
    if pd.isna(icd_code) or pd.isna(icd_version) or not range_min:
        return False

    # 预处理：移除 ICD-9/10 的小数点，只比较有效字符
    clean_code = str(icd_code).replace('.', '').upper()
    clean_min = str(range_min).replace('.', '').upper()
    clean_max = str(range_max).replace('.', '').upper()
    
    # 确保 ICD code 至少有 3 位前缀
    if len(clean_code) < 3:
        return False
        
    code_prefix = clean_code[:3] # 提取 3 位类别前缀
    
    if icd_version == 9:
        # ICD-9 使用 3 位前缀比较
        if len(clean_min) != 3 or len(clean_max) != 3:
            # 如果范围不是 3 位，尝试使用长度匹配
            code_prefix_match = clean_code[:len(clean_min)] 
            return code_prefix_match >= clean_min and code_prefix_match <= clean_max
        else:
            return code_prefix >= clean_min and code_prefix <= clean_max
        
    elif icd_version == 10:
        # ICD-10 保持不变 (基于 clean_min 长度的字符串比较)
        code_prefix_icd10 = clean_code[:len(clean_min)] 
        return code_prefix_icd10 >= clean_min and code_prefix_icd10 <= clean_max
    
    return False

def build_match_map(df):
    """将分类 DataFrame 转换为易于查找的匹配字典列表，处理多重范围和空值"""
    match_map = []
    for index, row in df.iterrows():
        internal_code = row['InternalCode']
        
        # --- 针对 ICD-10 Code ---
        icd10_code = row['ICD10_Code']
        if pd.notna(icd10_code):
            for part in str(icd10_code).split('/'):
                min_code, max_code = parse_single_icd_range(part.strip())
                if min_code:
                    match_map.append({'code': internal_code, 'version': 10, 'min': min_code, 'max': max_code})
        
        # --- 针对 ICD-9 Code ---
        icd9_code = row['ICD9_Code']
        if pd.notna(icd9_code): 
            for part in str(icd9_code).split('/'):
                min_code, max_code = parse_single_icd_range(part.strip())
                if min_code:
                    match_map.append({'code': internal_code, 'version': 9, 'min': min_code, 'max': max_code})
    return match_map

def read_report_text(relative_path, root_dir):
    """根据相对路径读取 CXR 报告文本。"""
    if pd.isna(relative_path) or relative_path == '':
        return None, False
    full_path = os.path.join(root_dir, relative_path)
    if not os.path.exists(full_path):
        return None, False 
    try:
        # CXR 报告位于 .../mimiciv/cxr/files/p**/p*******/s********.txt 目录下
        with open(full_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text, True 
    except Exception as e:
        return None, False 
# ================= 0. 死亡/再入院标签生成 (Step 0: Label Generation) =================

def step_0_generate_labels(run_mode):
    OUTPUT_DIR = PATHS['step0_output_dir']
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    log_filename = f'step_0_labels_{run_mode.lower()}.log'
    logger = setup_logger(OUTPUT_DIR, log_filename, 'Step0_Label_Gen')
    logger.info("="*60)
    logger.info(f"Starting Step 0: Label Generation in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*60)

    # ---------------------------------------------------------
    # 2. 加载基础数据 (Load Data)
    # ---------------------------------------------------------
    
    # >> Admissions (主表)
    adm_path = os.path.join(PATHS['hosp'], 'admissions.csv.gz')
    logger.info(f"Loading Admissions: {adm_path}")
    df_adm = pd.read_csv(adm_path, compression='gzip', 
                         usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'hospital_expire_flag'])
    
    # [DEBUG 过滤器]
    if run_mode == 'DEBUG':
        unique_subjects = df_adm['subject_id'].unique()[:100]
        logger.info(f"[DEBUG] Filtering for first 100 subjects.")
        df_adm = df_adm[df_adm['subject_id'].isin(unique_subjects)].copy()
    else:
        unique_subjects = None

    # 统计信息 (加入统计信息)
    num_adm = len(df_adm)
    num_subjects_adm = df_adm['subject_id'].nunique()
    logger.info(f"Loaded {num_adm} admissions from {num_subjects_adm} unique subjects.") # 加入统计信息

    # 时间转换
    for col in ['admittime', 'dischtime', 'deathtime']:
        df_adm[col] = pd.to_datetime(df_adm[col], errors='coerce')

    # >> Patients (DOD)
    logger.info("Loading Patients...")
    pat_path = os.path.join(PATHS['hosp'], 'patients.csv.gz')
    df_pat = pd.read_csv(pat_path, compression='gzip', usecols=['subject_id', 'dod'])
    df_pat['dod'] = pd.to_datetime(df_pat['dod'], errors='coerce')
    if run_mode == 'DEBUG' and unique_subjects is not None:
        df_pat = df_pat[df_pat['subject_id'].isin(unique_subjects)]

    # >> ICU Stays (用于验证)
    logger.info("Loading ICU Stays...")
    icu_path = os.path.join(PATHS['icu'], 'icustays.csv.gz')
    df_icu = pd.read_csv(icu_path, compression='gzip', usecols=['subject_id', 'hadm_id', 'stay_id', 'intime'])
    df_icu['intime'] = pd.to_datetime(df_icu['intime'], errors='coerce')
    df_icu_first = df_icu.sort_values(['subject_id', 'intime']).groupby('hadm_id').first().reset_index()

    # ---------------------------------------------------------
    # 3. 计算标签 (Calculation)
    # ---------------------------------------------------------
    logger.info("Calculating Mortality & Readmission...")
    
    # ... (标签计算逻辑不变) ...

    df_label = pd.merge(df_adm, df_pat, on='subject_id', how='left')
    df_icu_curr = df_icu_first[['hadm_id', 'stay_id']].rename(columns={'stay_id': 'curr_stay_id'})
    df_label = pd.merge(df_label, df_icu_curr, on='hadm_id', how='left')
    df_label.rename(columns={'hospital_expire_flag': 'mortality_in_hospital'}, inplace=True)
    df_label['final_death_date'] = df_label['deathtime'].combine_first(df_label['dod'])
    df_label['diff_death_days_raw'] = (df_label['final_death_date'] - df_label['dischtime']).dt.total_seconds() / (24*3600)
    df_label['mortality_30d'] = 0
    mask_dead_30d = (df_label['diff_death_days_raw'] > 0) & (df_label['diff_death_days_raw'] <= 30)
    df_label.loc[mask_dead_30d, 'mortality_30d'] = 1
    df_label = df_label.sort_values(by=['subject_id', 'admittime'])
    df_label['next_hadm_id'] = df_label.groupby('subject_id')['hadm_id'].shift(-1)
    df_label['next_admittime'] = df_label.groupby('subject_id')['admittime'].shift(-1)
    df_icu_next = df_icu_first[['hadm_id', 'stay_id']].rename(
        columns={'hadm_id': 'next_hadm_id_join_key', 'stay_id': 'next_stay_id'}
    )
    df_label = pd.merge(df_label, df_icu_next, left_on='next_hadm_id', right_on='next_hadm_id_join_key', how='left')
    df_label.drop(columns=['next_hadm_id_join_key'], inplace=True)
    df_label['diff_next_adm_days_raw'] = (df_label['next_admittime'] - df_label['dischtime']).dt.total_seconds() / (24*3600)
    df_label['readmission_30d_hosp'] = 0
    mask_readm_hosp = (df_label['diff_next_adm_days_raw'] <= 30) & (df_label['diff_next_adm_days_raw'] > 0)
    df_label.loc[mask_readm_hosp, 'readmission_30d_hosp'] = 1
    df_label['readmission_30d_icu'] = 0
    mask_readm_icu = (df_label['readmission_30d_hosp'] == 1) & (df_label['next_stay_id'].notna())
    df_label.loc[mask_readm_icu, 'readmission_30d_icu'] = 1
    df_label['days_disch_to_death'] = df_label['diff_death_days_raw'].round(0).astype('Int64')
    df_label['days_to_next_admission'] = df_label['diff_next_adm_days_raw'].round(0).astype('Int64')

    # ---------------------------------------------------------
    # 4. 关联 ICD 诊断
    # ---------------------------------------------------------
    logger.info("Mapping ICD Codes...")
    diag_path = os.path.join(PATHS['hosp'], 'diagnoses_icd.csv.gz')
    df_diag = pd.read_csv(diag_path, compression='gzip', dtype={'icd_code': str, 'icd_version': 'Int64'})
    if run_mode == 'DEBUG' and unique_subjects is not None:
        df_diag = df_diag[df_diag['subject_id'].isin(unique_subjects)]

    dict_path = os.path.join(PATHS['hosp'], 'd_icd_diagnoses.csv.gz')
    df_dict = pd.read_csv(dict_path, compression='gzip', usecols=['icd_code', 'icd_version', 'long_title'])

    # Merge
    df_final = pd.merge(df_diag, df_label, on=['subject_id', 'hadm_id'], how='left')
    df_final = pd.merge(df_final, df_dict, on=['icd_code', 'icd_version'], how='left')

    # 统计信息 (加入统计信息)
    num_diag_records = len(df_final)
    logger.info(f"Total ICD Diagnosis Records (Input for Step 1): {num_diag_records} records.")

    # ---------------------------------------------------------
    # 5. 保存文件 (Split & Save)
    # ---------------------------------------------------------
    
    # ... (保存逻辑不变) ...
    cols_labels = [
        'subject_id', 'hadm_id', 'seq_num', 
        'icd_code', 'icd_version', 'long_title',
        'mortality_in_hospital', 'mortality_30d', 
        'readmission_30d_hosp', 'readmission_30d_icu'
    ]
    
    cols_details = [
        'subject_id', 'hadm_id', 'seq_num', 
        'icd_code', 'icd_version', 'long_title',
        'mortality_in_hospital', 'mortality_30d', 
        'readmission_30d_hosp', 'readmission_30d_icu',
        'admittime', 'dischtime',
        'dod', 'final_death_date', 'days_disch_to_death',
        'curr_stay_id', 
        'next_hadm_id', 'next_admittime', 'days_to_next_admission', 'next_stay_id'
    ]
    
    suffix = run_mode.lower()

    # 1. 保存 Labels 表
    filename_labels = f'mimiciv_3_1_labels_{suffix}.csv.gz'
    path_labels = os.path.join(OUTPUT_DIR, filename_labels)
    
    logger.info(f"Saving [Labels] table to: {path_labels}")
    valid_cols_labels = [c for c in cols_labels if c in df_final.columns]
    df_final[valid_cols_labels].to_csv(path_labels, index=False, compression='gzip')

    # 2. 保存 Details 表 (用于下一步 CVD 过滤的输入)
    filename_details = f'mimiciv_3_1_labels_details_{suffix}.csv.gz'
    path_details = os.path.join(OUTPUT_DIR, filename_details)
    
    logger.info(f"Saving [Details] table to: {path_details}")
    valid_cols_details = [c for c in cols_details if c in df_final.columns]
    df_final[valid_cols_details].to_csv(path_details, index=False, compression='gzip')

    logger.info("Step 0 completed successfully. Returning Details DataFrame for Step 1.")
    
    return df_final[valid_cols_details].copy(), True


# ================= 1. CVD 标签匹配和过滤 (Step 1: Labeling & Filtering) =================

def step_1_cvd_labeling(run_mode):
    suffix = run_mode.lower()
    OUTPUT_DIR = PATHS['step1_output_dir']
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    log_filename = f'step_1_cvd_labeling_{suffix}.log' 
    logger = setup_logger(OUTPUT_DIR, log_filename, 'Step1_CVD_Labeler')
    logger.info("="*60)
    logger.info(f"Starting Step 1: CVD Labeling & Filtering in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # 确定输入文件: 
    input_suffix = run_mode.lower()
    input_filename = f'mimiciv_3_1_labels_details_{input_suffix}.csv.gz'
    input_path = os.path.join(PATHS['step0_output_dir'], input_filename)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file from Step 0 not found: {input_path}")
        return pd.DataFrame(), False

    # 读取 Step 0 的输出
    df_input = pd.read_csv(input_path, compression='gzip', 
                           dtype={'icd_code': str, 'icd_version': 'Int64'}) 

    # 统计信息 (加入统计信息)
    num_input_records = len(df_input)
    num_input_adm = df_input['hadm_id'].nunique()
    num_input_subjects = df_input['subject_id'].nunique()
    logger.info(f"Input records from Step 0: {num_input_records} records, {num_input_adm} admissions, {num_input_subjects} subjects.")
    
    # ... (加载分类文件和匹配逻辑不变) ...
    coarse_cat_path = os.path.join(PATHS['CVD_CATEGORY_PATH'], 'CVD_coarse_category.csv')
    fine_cat_path = os.path.join(PATHS['CVD_CATEGORY_PATH'], 'CVD_fine_category.csv')
    if not os.path.exists(coarse_cat_path) or not os.path.exists(fine_cat_path):
        logger.error(f"Category files not found. Check path: {PATHS['CVD_CATEGORY_PATH']}")
        return pd.DataFrame(), False

    df_coarse = pd.read_csv(coarse_cat_path)
    df_fine = pd.read_csv(fine_cat_path)

    # 3. 构建多重匹配规则
    coarse_map = build_match_map(df_coarse)
    fine_map = build_match_map(df_fine)
    
    # 4. CVD 标签匹配
    df_input['CVD_coarse_category'] = pd.NA
    df_input['CVD_fine_category'] = pd.NA
    
    def apply_cvd_matching(row):
        icd_code = row['icd_code']
        version = row['icd_version']
        coarse_cat = pd.NA
        fine_cat = pd.NA
        if pd.isna(icd_code):
            return pd.Series([pd.NA, pd.NA], index=['CVD_coarse_category', 'CVD_fine_category'])
            
        # 1. 匹配 Fine Category
        for item in fine_map:
            if item['version'] == version and is_icd_in_range(icd_code, version, item['min'], item['max']):
                fine_cat = item['code']
                break
        
        # 2. 匹配 Coarse Category
        for item in coarse_map:
            if item['version'] == version and is_icd_in_range(icd_code, version, item['min'], item['max']):
                coarse_cat = item['code']
                break
        return pd.Series([coarse_cat, fine_cat], index=['CVD_coarse_category', 'CVD_fine_category'])

    logger.info("Applying CVD matching function...")
    time_cols = ['admittime', 'dischtime']
    for col in time_cols:
        df_input[col] = pd.to_datetime(df_input[col], errors='coerce')
        
    df_input[['CVD_coarse_category', 'CVD_fine_category']] = df_input.progress_apply(apply_cvd_matching, axis=1)

    # 包含 CVD 匹配记录的子集 (用于后续 MM 匹配的基础队列)
    df_cvd_only_for_mm = df_input[df_input['CVD_coarse_category'].notna()].copy()
    
    # 统计信息 (加入统计信息)
    num_cvd_records = len(df_cvd_only_for_mm)
    num_cvd_adm = df_cvd_only_for_mm['hadm_id'].nunique()
    num_cvd_subjects = df_cvd_only_for_mm['subject_id'].nunique()
    logger.info(f"Total ICD records matched CVD categories: {num_cvd_records} records.")
    logger.info(f"CVD Cohort Statistics: {num_cvd_adm} unique admissions, {num_cvd_subjects} unique subjects.")
    
    # 5. 保存输出文件
    filename_cvd_only_details = f'step_1_details_cvd_only_{suffix}.csv.gz' 
    path_cvd_only_details = os.path.join(OUTPUT_DIR, filename_cvd_only_details)

    logger.info(f"Saving [CVD ONLY, Details Style] to: {path_cvd_only_details}")
    df_cvd_only_for_mm.to_csv(path_cvd_only_details, index=False, compression='gzip')
    
    logger.info("Step 1 completed successfully.")
    
    return df_cvd_only_for_mm, True

# ================= 2A. Note 匹配阶段 (Step 2A: Note Matching) =================

def step_2a_note_matching(run_mode, df_cvd_input):
    suffix = run_mode.lower()
    OUTPUT_DIR = PATHS['step2_output_dir']
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    logger = setup_logger(OUTPUT_DIR, f'step_2a_match_notes_{suffix}.log', 'Step2A_Note_Processor')
    logger.info("="*60)
    logger.info(f"Starting Step 2A: Note Matching Pipeline in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    df_cvd = df_cvd_input.copy()
    valid_subjects = set(df_cvd['subject_id'].unique())
    
    # 统计信息 (加入统计信息)
    num_input_records = len(df_cvd)
    num_input_adm = df_cvd['hadm_id'].nunique()
    num_input_subjects = df_cvd['subject_id'].nunique()
    logger.info(f"Input CVD Cohort size: {num_input_records} records, {num_input_adm} admissions, {num_input_subjects} subjects.")

    # ... (时间索引构建逻辑不变) ...
    subject_stay_map = {}
    df_time_index = df_cvd[['subject_id', 'hadm_id', 'admittime', 'dischtime']].drop_duplicates()
    def build_index(row):
        sid, hid, start, end = row['subject_id'], row['hadm_id'], row['admittime'], row['dischtime']
        if pd.isna(start) or pd.isna(end): return 
        if sid not in subject_stay_map: subject_stay_map[sid] = []
        subject_stay_map[sid].append({'hadm_id': hid, 'start': start, 'end': end})
        
    logger.info("Building time index...")
    df_time_index.progress_apply(build_index, axis=1)

    # 3. 流式处理 Note 文件并匹配
    matched_notes_list = []
    match_cols = ['note_id', 'subject_id', 'hadm_id', 'note_type', 'note_seq', 'charttime', 'storetime', 'text']

    for note_type, filename in NOTE_FILES.items():
        # ... (Note 文件读取和匹配逻辑不变) ...
        note_path = os.path.join(PATHS['note_dir'], filename)
        if not os.path.exists(note_path):
            logger.warning(f"Note file not found: {note_path}. Skipping.")
            continue
        chunk_size = 100000 
        chunks_to_process = 6 if run_mode == 'DEBUG' else float('inf')

        with pd.read_csv(note_path, compression='gzip', usecols=match_cols, chunksize=chunk_size, 
                         dtype={'note_id': str, 'subject_id': 'Int64', 'hadm_id': 'Int64', 'note_seq': 'Int64'}) as reader:
            for i, chunk in tqdm(enumerate(reader), desc=f"Processing {note_type} Chunks"):
                
                chunk_filtered = chunk[chunk['subject_id'].isin(valid_subjects)].copy()
                
                if not chunk_filtered.empty:
                    chunk_filtered['charttime'] = pd.to_datetime(chunk_filtered['charttime'], errors='coerce')
                    
                    def find_matching_hadm(row):
                        sid, ctime, current_hadm = row['subject_id'], row['charttime'], row['hadm_id']
                        possible_stays = subject_stay_map.get(sid, [])
                        
                        # 1. 优先使用 charttime 匹配
                        for stay in possible_stays:
                            if pd.notna(ctime) and stay['start'] <= ctime <= stay['end']:
                                return stay['hadm_id']
                        
                        # 2. 如果 charttime 不可用，且 note 有 hadm_id，尝试用该 hadm_id 确认是否在 cohort 中
                        if pd.notna(current_hadm):
                            for stay in possible_stays:
                                if stay['hadm_id'] == current_hadm:
                                    return current_hadm
                        return None

                    chunk_filtered['matched_hadm_id'] = chunk_filtered.progress_apply(find_matching_hadm, axis=1)
                    final_matches = chunk_filtered[chunk_filtered['matched_hadm_id'].notna()].copy()
                    
                    if not final_matches.empty:
                        final_matches['hadm_id'] = final_matches['matched_hadm_id']
                        final_matches.drop(columns=['matched_hadm_id'], inplace=True)
                        matched_notes_list.append(final_matches)

                if run_mode == 'DEBUG' and i >= chunks_to_process: 
                    logger.info(f"DEBUG mode: Processed {i+1} chunks of {filename}. Breaking loop.")
                    break

    # 4. 整合结果
    if not matched_notes_list:
        logger.warning("No matching notes found!")
        df_merged = df_cvd.copy()
        for col in ['has_note', 'note_count', 'matched_note_ids', 'matched_note_types', 'matched_note_times']:
             df_merged[col] = 0 if col in ['has_note', 'note_count'] else df_merged.apply(lambda x: [], axis=1)
    else:
        df_all_notes = pd.concat(matched_notes_list, ignore_index=True)
        # 统计信息 (加入统计信息)
        num_matched_notes = len(df_all_notes)
        num_matched_adms = df_all_notes['hadm_id'].nunique()
        logger.info(f"Total matched notes found: {num_matched_notes} records, matched across {num_matched_adms} admissions.") 

        # ... (保存 Note Content 和聚合逻辑不变) ...
        output_notes_filename = f'cvd_matched_notes_content_{suffix}.csv.gz'
        cols_note_content = ['subject_id', 'hadm_id', 'note_id', 'note_type', 'charttime', 'storetime', 'text']
        df_all_notes[cols_note_content].to_csv(os.path.join(OUTPUT_DIR, output_notes_filename), index=False, compression='gzip')
        logger.info(f"Saved matched Note Content (Text): {output_notes_filename}")

        note_stats = df_all_notes.groupby('hadm_id').agg({
            'note_id': list, 
            'note_type': lambda x: list(set(x.dropna().astype(str))), 
            'charttime': list
        }).reset_index()
        note_stats['note_count'] = note_stats['note_id'].apply(len)
        note_stats['has_note'] = 1
        note_stats.rename(columns={'note_id': 'matched_note_ids', 'note_type': 'matched_note_types', 'charttime': 'matched_note_times'}, inplace=True)
        df_merged = pd.merge(df_cvd, note_stats, on='hadm_id', how='left')
        df_merged['has_note'] = df_merged['has_note'].fillna(0).astype(int)
        df_merged['note_count'] = df_merged['note_count'].fillna(0).astype(int)
        for col in ['matched_note_ids', 'matched_note_types', 'matched_note_times']:
            df_merged[col] = df_merged[col].apply(lambda x: x if isinstance(x, list) else [])

    output_details_file = f'step_2a_details_cvd_with_note_{suffix}.csv.gz'
    df_merged.to_csv(os.path.join(OUTPUT_DIR, output_details_file), index=False, compression='gzip')
    
    logger.info("Step 2A: Note Pipeline completed successfully.")
    return df_merged

# ================= 2B. CXR 匹配阶段 (Step 2B: CXR Matching) =================

def step_2b_cxr_matching(run_mode, df_cvd_note_input):
    suffix = run_mode.lower()
    OUTPUT_DIR = PATHS['step2_output_dir']
    logger = setup_logger(OUTPUT_DIR, f'step_2b_match_cxr_{suffix}.log', 'Step2B_CXR_Processor')
    logger.info("="*60)
    logger.info(f"Starting Step 2B: CXR Matching Pipeline in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    df_cvd = df_cvd_note_input.copy()
    valid_subjects = set(df_cvd['subject_id'].unique())
    
    # 统计信息 (加入统计信息)
    num_input_records = len(df_cvd)
    num_input_adm = df_cvd['hadm_id'].nunique()
    num_input_subjects = df_cvd['subject_id'].nunique()
    logger.info(f"Input CVD Cohort size: {num_input_records} records, {num_input_adm} admissions, {num_input_subjects} subjects.")
    
    # 2. 加载并预处理 CXR Metadata
    # ... (元数据加载和时间解析逻辑不变) ...
    meta_path = os.path.join(PATHS['cxr_dir'], CXR_FILES['metadata'])
    df_meta = pd.read_csv(meta_path, compression='gzip', 
                          usecols=['subject_id', 'study_id', 'dicom_id', 'StudyDate', 'StudyTime', 'ViewPosition'],
                          dtype={'subject_id': 'Int64', 'study_id': 'Int64'})
    df_meta = df_meta[df_meta['subject_id'].isin(valid_subjects)].copy()

    def parse_dicom_date_time(row):
        d = str(row['StudyDate'])
        t = str(row['StudyTime']).split('.')[0] 
        if len(t) < 6: t = t.zfill(6) 
        dt_str = f"{d} {t}"
        return pd.to_datetime(dt_str, format='%Y%m%d %H%M%S', errors='coerce')

    logger.info("Parsing CXR timestamps...")
    df_meta['cxr_time'] = df_meta.progress_apply(parse_dicom_date_time, axis=1)
    
    # 3. 执行时间窗口匹配
    df_cvd_keys = df_cvd[['subject_id', 'hadm_id', 'admittime', 'dischtime']].drop_duplicates()
    df_merged_time = pd.merge(df_meta, df_cvd_keys, on='subject_id', how='inner')
    mask_time = (df_merged_time['cxr_time'] >= df_merged_time['admittime']) & (df_merged_time['cxr_time'] <= df_merged_time['dischtime'])
    matched_cxr = df_merged_time[mask_time].copy()
    
    # 统计信息 (加入统计信息)
    num_matched_images = len(matched_cxr)
    num_matched_adms_images = matched_cxr['hadm_id'].nunique()
    logger.info(f"Matched {num_matched_images} CXR images across {num_matched_adms_images} admissions (Time Window Match).")
    
    # 4. 关联 CheXpert 标签和路径信息
    # ... (关联逻辑不变) ...
    df_chexpert = pd.read_csv(os.path.join(PATHS['cxr_dir'], CXR_FILES['chexpert']), compression='gzip', 
                              dtype={'subject_id': 'Int64', 'study_id': 'Int64'})
    df_record = pd.read_csv(os.path.join(PATHS['cxr_dir'], CXR_FILES['record_list']), compression='gzip', 
                            dtype={'subject_id': 'Int64', 'study_id': 'Int64'})
    df_record.rename(columns={'path': 'cxr_image_path_relative'}, inplace=True)
    df_study_list = pd.read_csv(os.path.join(PATHS['cxr_dir'], CXR_FILES['study_list']), compression='gzip', 
                                dtype={'subject_id': 'Int64', 'study_id': 'Int64'})
    df_study_list.rename(columns={'path': 'cxr_report_path_relative'}, inplace=True)

    matched_full = pd.merge(matched_cxr, df_record[['dicom_id', 'cxr_image_path_relative']], on='dicom_id', how='left')
    matched_full = pd.merge(matched_full, df_chexpert, on=['subject_id', 'study_id'], how='left')
    matched_full = pd.merge(matched_full, df_study_list, on=['subject_id', 'study_id'], how='left')
    matched_full.drop(columns=['admittime', 'dischtime'], inplace=True) 

    # 5. 提取报告文本 (保存到单独文件)
    unique_studies = matched_full[['subject_id', 'hadm_id', 'study_id', 'cxr_report_path_relative', 'cxr_time']].drop_duplicates()
    
    # 统计信息 (加入统计信息)
    num_matched_studies = len(unique_studies)
    logger.info(f"Total unique CXR studies found for report extraction: {num_matched_studies}.")
    
    if run_mode == 'DEBUG' and len(unique_studies) > 20: unique_studies = unique_studies.head(20)
    
    results = []
    for idx, row in tqdm(unique_studies.iterrows(), total=len(unique_studies), desc="Reading CXR Reports"):
        text, success = read_report_text(row['cxr_report_path_relative'], PATHS['cxr_reports_root'])
        results.append({'subject_id': row['subject_id'], 'hadm_id': row['hadm_id'], 'study_id': row['study_id'], 
                        'cxr_report_path_relative': row['cxr_report_path_relative'], 
                        'cxr_study_time': row['cxr_time'], 'report_text': text})
    
    df_reports_save = pd.DataFrame(results)
    
    # ... (CXR Reports/Metadata 保存逻辑不变) ...
    df_reports_save.rename(columns={
        'cxr_report_path_relative': 'cxr_report_path',
        'cxr_study_time': 'cxr_time'
    }, inplace=True)
    cxr_report_cols_save = ['subject_id', 'hadm_id', 'study_id', 'cxr_time', 'report_text', 'cxr_report_path'] 
    cxr_report_cols_save = [c for c in cxr_report_cols_save if c in df_reports_save.columns] 

    output_reports_filename = f'cvd_matched_cxr_reports_{suffix}.csv.gz' 
    df_reports_save[cxr_report_cols_save].to_csv(os.path.join(OUTPUT_DIR, output_reports_filename), index=False, compression='gzip')
    logger.info(f"Saved matched CXR Reports (Text/Path): {output_reports_filename}")

    cols_cxr_metadata = ['subject_id', 'hadm_id', 'study_id', 'dicom_id', 'cxr_time', 'ViewPosition', 
                         'cxr_image_path_relative'] + [f for f in CXR_CHEXPERT_LABELS if f in matched_full.columns]
    
    output_metadata_filename = f'cvd_matched_cxr_metadata_{suffix}.csv.gz' 
    matched_full[cols_cxr_metadata].to_csv(os.path.join(OUTPUT_DIR, output_metadata_filename), index=False, compression='gzip')
    logger.info(f"Saved matched CXR Metadata/CheXpert: {output_metadata_filename}")


    # 6. 聚合 CXR 详细信息到列表 (HADm 级别) 
    # ... (聚合逻辑不变) ...
    aggregation_cols = {
        'study_id': ('matched_cxr_study_ids', lambda x: list(set(x.dropna().astype('Int64')))), 
        'dicom_id': ('matched_cxr_dicom_ids', lambda x: list(x.dropna())), 
        'cxr_time': ('matched_cxr_times', lambda x: list(x.dropna().astype(str))), 
        'ViewPosition': ('matched_cxr_positions', lambda x: list(x.dropna().astype(str))),     
        'cxr_image_path_relative': ('matched_cxr_image_paths', lambda x: list(x.dropna().astype(str))), 
        'cxr_report_path_relative': ('matched_cxr_report_paths', lambda x: list(x.dropna().astype(str)))
    }
    for label in CXR_CHEXPERT_LABELS:
        if label in matched_full.columns: 
            aggregation_cols[label] = (f'matched_cxr_{label}', lambda x: list(x.dropna().astype(int)))
            
    agg_dict = {col_in: agg_func for col_in, (col_out, agg_func) in aggregation_cols.items()}
    df_cxr_lists = matched_full.groupby('hadm_id').agg(agg_dict).reset_index()
    df_cxr_lists.rename(columns={col_in: col_out for col_in, (col_out, agg_func) in aggregation_cols.items()}, inplace=True)
    
    df_cxr_lists['cxr_image_count'] = df_cxr_lists['matched_cxr_dicom_ids'].apply(len) if 'matched_cxr_dicom_ids' in df_cxr_lists.columns else 0
    df_cxr_lists['cxr_study_count'] = df_cxr_lists['matched_cxr_study_ids'].apply(len) if 'matched_cxr_study_ids' in df_cxr_lists.columns else 0
    df_cxr_lists['has_cxr'] = 1

    # 7. 最终合并与保存
    df_final = pd.merge(df_cvd, df_cxr_lists, on='hadm_id', how='left')
    df_final['has_cxr'] = df_final['has_cxr'].fillna(0).astype(int)
    df_final['cxr_study_count'] = df_final['cxr_study_count'].fillna(0).astype(int)
    df_final['cxr_image_count'] = df_final['cxr_image_count'].fillna(0).astype(int)
    
    list_cols = [c for c in df_cxr_lists.columns if c not in ['hadm_id', 'has_cxr', 'cxr_study_count', 'cxr_image_count']]
    for col in list_cols: df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, list) else [])

    df_final.to_csv(os.path.join(OUTPUT_DIR, f'step_2b_details_cvd_with_note_with_cxr_{suffix}.csv.gz'), index=False, compression='gzip')
    logger.info("Step 2B: CXR Pipeline completed successfully.")
    
    return df_final 

# ================= 2C. ECG 匹配阶段 (Step 2C: ECG Matching) =================

def step_2c_ecg_matching(run_mode, df_cvd_cxr_input):
    suffix = run_mode.lower()
    OUTPUT_DIR = PATHS['step2_output_dir']
    logger = setup_logger(OUTPUT_DIR, f'step_2c_match_ecg_{suffix}.log', 'Step2C_ECG_Processor')
    logger.info("="*60)
    logger.info(f"Starting Step 2C: ECG Matching Pipeline in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    df_cvd_cxr = df_cvd_cxr_input.copy()
    valid_subjects = set(df_cvd_cxr['subject_id'].unique())
    
    # 统计信息 (加入统计信息)
    num_input_records = len(df_cvd_cxr)
    num_input_adm = df_cvd_cxr['hadm_id'].nunique()
    num_input_subjects = df_cvd_cxr['subject_id'].nunique()
    logger.info(f"Input CVD Cohort size: {num_input_records} records, {num_input_adm} admissions, {num_input_subjects} subjects.")

    # 2. 加载 ECG 元数据
    # ************************************************
    # *** 修正 ValueError: 根据用户提供的实际列头进行修正 ***
    # ************************************************
    measurements_path = os.path.join(PATHS['ecg_dir'], ECG_FILES['machine_measurements'])
    
    # 移除缺失的列，并加入 'bandwidth' 和 'filtering'
    MEASUREMENT_COLS_LOAD = [
        'subject_id', 'study_id', 'cart_id', 'ecg_time', 
        'report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 
        'report_6', 'report_7', 'report_8', 'report_9', 'report_10', 'report_11', 
        'report_12', 'report_13', 'report_14', 'report_15', 'report_16', 'report_17',
        'bandwidth', 'filtering', # <--- 根据用户提供的目录修正
        'rr_interval', 
        'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 
        'p_axis', 'qrs_axis', 't_axis'
    ]

    MEASUREMENT_DTYPE_SPEC = {
        'subject_id': 'Int64', 'study_id': 'Int64', 'cart_id': 'Int64', 'ecg_time': str,
        'bandwidth': str, 'filtering': str, # <--- 根据用户提供的目录修正
        'rr_interval': 'Int64', 
        'p_onset': 'Int64', 'p_end': 'Int64', 'qrs_onset': 'Int64', 'qrs_end': 'Int64', 't_end': 'Int64',
        'p_axis': 'Int64', 'qrs_axis': 'Int64', 't_axis': 'Int64'
    }
    for i in range(18): MEASUREMENT_DTYPE_SPEC[f'report_{i}'] = str 
    
    df_measurements = pd.read_csv(measurements_path, usecols=MEASUREMENT_COLS_LOAD, 
                                  dtype=MEASUREMENT_DTYPE_SPEC, low_memory=False)  
    # ************************************************
    
    df_measurements = df_measurements[df_measurements['subject_id'].isin(valid_subjects)].copy()
    df_measurements['ecg_time'] = pd.to_datetime(df_measurements['ecg_time'], errors='coerce')
    df_measurements.dropna(subset=['ecg_time'], inplace=True)
    
    record_list_path = os.path.join(PATHS['ecg_dir'], ECG_FILES['record_list'])
    # 注意: record_list.csv 的列头没有提供，但根据 MIMIC 惯例，它通常包含 subject_id, study_id, path
    # 这里我们假设它与之前版本的一致，但用户提供的目录中并未明确说明 record_list.csv 的列头，
    # 我们基于 waveform_note_links.csv 提供的 study_id 和 subject_id 来推断 record_list.csv 的内容。
    # 鉴于代码中使用了 'path' 列，我们假设它存在且是正确的路径列。
    # 如果您收到 record_list.csv 的错误，可能需要修改此处的 usecols。
    df_record = pd.read_csv(record_list_path, usecols=['subject_id', 'study_id', 'path'],
                            dtype={'subject_id': 'Int64', 'study_id': 'Int64'})
    df_record.rename(columns={'path': 'ecg_waveform_path_relative'}, inplace=True)
    
    note_links_path = os.path.join(PATHS['ecg_dir'], ECG_FILES['waveform_note_links'])
    # 根据用户提供的 waveform_note_links.csv 目录进行修正
    DTYPE_SPEC_NOTE = {'subject_id': 'Int64', 'study_id': 'Int64', 'note_id': str, 'note_seq': 'Int64', 'charttime': str}
    # 根据用户提供的目录，waveform_note_links.csv 包含 subject_id study_id note_id note_seq charttime
    df_note_links = pd.read_csv(note_links_path, usecols=['subject_id', 'study_id', 'note_id', 'note_seq', 'charttime'],
                                dtype=DTYPE_SPEC_NOTE) 
    df_note_links['charttime'] = pd.to_datetime(df_note_links['charttime'], errors='coerce')

    # 3. 执行时间窗口匹配
    df_cvd_keys = df_cvd_cxr[['subject_id', 'hadm_id', 'admittime', 'dischtime']].drop_duplicates()
    df_merged = pd.merge(df_measurements, df_cvd_keys, on='subject_id', how='inner')
    mask_time = (df_merged['ecg_time'] >= df_merged['admittime']) & (df_merged['ecg_time'] <= df_merged['dischtime'])
    matched_ecg = df_merged[mask_time].copy()
    
    # 统计信息 (加入统计信息)
    num_matched_ecg_records = len(matched_ecg)
    num_matched_adms = matched_ecg['hadm_id'].nunique()
    logger.info(f"Matched {num_matched_ecg_records} ECG records (measurements) across {num_matched_adms} admissions.")

    # 4. 关联路径和 Note 链接
    matched_full = pd.merge(matched_ecg, df_record[['subject_id', 'study_id', 'ecg_waveform_path_relative']], on=['subject_id', 'study_id'], how='left')
    matched_full = pd.merge(matched_full, df_note_links[['subject_id', 'study_id', 'note_id', 'note_seq', 'charttime']], on=['subject_id', 'study_id'], how='left')
    matched_full.drop(columns=['admittime', 'dischtime'], inplace=True) 

    # -------------------------------------------------------------
    # 5. 保存 ECG Measurements 文件
    # -------------------------------------------------------------
    
    # 修正保存列表，匹配已加载的列，并加入 'bandwidth', 'filtering'
    ECG_MEASUREMENT_COLS_USER_ORDER = [
        'subject_id', 'study_id', 'cart_id', 'ecg_time'
    ] + [f'report_{i}' for i in range(18)] + [
        'bandwidth', 'filtering', # <--- 修正
        'rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 
        'p_axis', 'qrs_axis', 't_axis', 'hadm_id'
    ]
    
    df_ecg_measurements_save = matched_full.copy()
    ecg_output_cols_user = [col for col in ECG_MEASUREMENT_COLS_USER_ORDER if col in df_ecg_measurements_save.columns]
    
    output_measurements_filename = f'cvd_matched_ecg_measurements_{suffix}.csv.gz' 
    df_ecg_measurements_save[ecg_output_cols_user].to_csv(os.path.join(OUTPUT_DIR, output_measurements_filename), index=False, compression='gzip')
    logger.info(f"Saved matched ECG Measurements: {output_measurements_filename}")

    # -------------------------------------------------------------
    # 6. 保存 ECG Details 文件
    # -------------------------------------------------------------
    
    # 修正保存列表，匹配已加载的列，并加入 'bandwidth', 'filtering'
    BASE_ECG_COLS_SAVE_FULL = (
        ['subject_id', 'hadm_id', 'study_id', 'cart_id', 'ecg_time', 'ecg_waveform_path_relative', 
         'note_id', 'note_seq', 'charttime'] +
        [f'report_{i}' for i in range(18)] + 
        ['bandwidth', 'filtering'] + # <--- 修正
        ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']
    )

    existing_base_cols_full = [col for col in BASE_ECG_COLS_SAVE_FULL if col in matched_full.columns]
    df_ecg_output = matched_full.copy()
    ecg_output_cols_details = existing_base_cols_full[:] 

    note_content_path = os.path.join(OUTPUT_DIR, f'cvd_matched_notes_content_{suffix}.csv.gz') 
    if os.path.exists(note_content_path):
        df_note_content = pd.read_csv(note_content_path, compression='gzip', 
                                      usecols=['note_id', 'text']).rename(columns={'text': 'ecg_note_text'})
        df_note_content = df_note_content.drop_duplicates(subset=['note_id']) 
        df_ecg_output = pd.merge(df_ecg_output, df_note_content, on='note_id', how='left')
        if 'ecg_note_text' in df_ecg_output.columns and 'ecg_note_text' not in ecg_output_cols_details:
            ecg_output_cols_details.append('ecg_note_text') 

    output_details_filename = f'cvd_matched_ecg_details_{suffix}.csv.gz' 
    df_ecg_output[ecg_output_cols_details].to_csv(os.path.join(OUTPUT_DIR, output_details_filename), index=False, compression='gzip')
    logger.info(f"Saved matched ECG Details (Measurements/Path/Note): {output_details_filename}")

    # -------------------------------------------------------------
    # 7. 聚合 ECG 详细信息到列表 (HADm 级别)
    # -------------------------------------------------------------
    ecg_agg_cols_potential = {
        'study_id': ('matched_ecg_study_ids', lambda x: list(set(x.dropna().astype('Int64')))), 
        'cart_id': ('matched_ecg_cart_ids', lambda x: list(x.dropna().astype('Int64'))),
        'ecg_time': ('matched_ecg_times', lambda x: list(x.dropna().astype(str))), 
        'ecg_waveform_path_relative': ('matched_ecg_waveform_paths', lambda x: list(x.dropna().astype(str))),
        'note_id': ('matched_ecg_mimic_note_ids', lambda x: list(set(x.dropna().astype(str)))),  
        'charttime': ('matched_ecg_note_charttimes', lambda x: list(set(x.dropna().astype(str)))),
        'bandwidth': ('matched_ecg_bandwidths', lambda x: list(x.dropna().astype(str))), # <--- 修正
        'filtering': ('matched_ecg_filterings', lambda x: list(x.dropna().astype(str))), # <--- 修正
        'rr_interval': ('matched_ecg_rr_intervals', lambda x: list(x.dropna().astype('Int64'))), 
        'p_onset': ('matched_ecg_p_onsets', lambda x: list(x.dropna().astype('Int64'))),
        'p_end': ('matched_ecg_p_ends', lambda x: list(x.dropna().astype('Int64'))), 
        'qrs_onset': ('matched_ecg_qrs_onsets', lambda x: list(x.dropna().astype('Int64'))), 
        'qrs_end': ('matched_ecg_qrs_ends', lambda x: list(x.dropna().astype('Int64'))),
        't_end': ('matched_ecg_t_ends', lambda x: list(x.dropna().astype('Int64'))),
        'p_axis': ('matched_ecg_p_axis', lambda x: list(x.dropna().astype('Int64'))), 
        'qrs_axis': ('matched_ecg_qrs_axis', lambda x: list(x.dropna().astype('Int64'))),
        't_axis': ('matched_ecg_t_axis', lambda x: list(x.dropna().astype('Int64')))
    }
    for i in range(18):
        col = f'report_{i}'
        ecg_agg_cols_potential[col] = (f'matched_ecg_{col}', lambda x: list(x.dropna().astype(str)))

    agg_dict = {col_in: agg_func for col_in, (col_out, agg_func) in ecg_agg_cols_potential.items() if col_in in matched_full.columns}
    df_ecg_lists = matched_full.groupby('hadm_id').agg(agg_dict).reset_index()
    rename_map = {col_in: col_out for col_in, (col_out, agg_func) in ecg_agg_cols_potential.items() if col_in in matched_full.columns}
    df_ecg_lists.rename(columns=rename_map, inplace=True)
    
    df_ecg_lists['ecg_count'] = df_ecg_lists['matched_ecg_cart_ids'].apply(len) if 'matched_ecg_cart_ids' in df_ecg_lists.columns else 0
    df_ecg_lists['has_ecg'] = 1

    df_final = pd.merge(df_cvd_cxr, df_ecg_lists, on='hadm_id', how='left')
    df_final['has_ecg'] = df_final['has_ecg'].fillna(0).astype(int)
    df_final['ecg_count'] = df_final['ecg_count'].fillna(0).astype(int)
    list_cols = [c for c in df_ecg_lists.columns if c not in ['hadm_id', 'has_ecg', 'ecg_count']]
    for col in list_cols: df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, list) else [])

    df_final.to_csv(os.path.join(OUTPUT_DIR, f'step_2c_details_cvd_with_note_with_cxr_with_ecg_{suffix}.csv.gz'), index=False, compression='gzip')
    logger.info("Step 2C: ECG Pipeline completed successfully.")
    
    return df_final 

# ================= 2D. Echo 匹配阶段 (Step 2D: Echo Matching) =================

def step_2d_echo_matching(run_mode, df_cvd_ecg_input):
    suffix = run_mode.lower()
    OUTPUT_DIR = PATHS['step2_output_dir']
    logger = setup_logger(OUTPUT_DIR, f'step_2d_match_echo_{suffix}.log', 'Step2D_Echo_Processor')
    logger.info("="*60)
    logger.info(f"Starting Step 2D: Echo Matching Pipeline in [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    df_cvd_ecg = df_cvd_ecg_input.copy()
    valid_subjects = set(df_cvd_ecg['subject_id'].unique())
    
    # 统计信息 (加入统计信息)
    num_input_records = len(df_cvd_ecg)
    num_input_adm = df_cvd_ecg['hadm_id'].nunique()
    num_input_subjects = df_cvd_ecg['subject_id'].nunique()
    logger.info(f"Input CVD Cohort size: {num_input_records} records, {num_input_adm} admissions, {num_input_subjects} subjects.")
    
    # 2. 加载 Echo 元数据
    # ... (Echo 数据加载和时间解析逻辑不变) ...
    study_list_path = os.path.join(PATHS['echo_dir'], ECHO_FILES['study_list'])
    DTYPE_SPEC = {'subject_id': 'Int64', 'study_id': 'Int64', 'note_id': str, 'study_datetime': str, 'note_charttime': str}
    df_study = pd.read_csv(study_list_path, dtype=DTYPE_SPEC) 
    df_study = df_study[df_study['subject_id'].isin(valid_subjects)].copy()
    df_study['echo_time'] = pd.to_datetime(df_study['study_datetime'], errors='coerce')
    df_study['note_charttime'] = pd.to_datetime(df_study['note_charttime'], errors='coerce')
    df_study.dropna(subset=['echo_time'], inplace=True)
    
    record_list_path = os.path.join(PATHS['echo_dir'], ECHO_FILES['record_list'])
    df_record = pd.read_csv(record_list_path, usecols=['subject_id', 'study_id', 'dicom_filepath'],
                            dtype={'subject_id': 'Int64', 'study_id': 'Int64'}) 
    df_record.rename(columns={'dicom_filepath': 'echo_dicom_path_relative'}, inplace=True)
    
    # 3. 执行时间窗口匹配
    df_cvd_keys = df_cvd_ecg[['subject_id', 'hadm_id', 'admittime', 'dischtime']].drop_duplicates()
    df_merged = pd.merge(df_study, df_cvd_keys, on='subject_id', how='inner')
    mask_time = (df_merged['echo_time'] >= df_merged['admittime']) & (df_merged['echo_time'] <= df_merged['dischtime'])
    matched_echo = df_merged[mask_time].copy()
    
    # 统计信息 (加入统计信息)
    num_matched_studies = len(matched_echo)
    num_matched_adms = matched_echo['hadm_id'].nunique()
    logger.info(f"Matched {num_matched_studies} Echo studies across {num_matched_adms} admissions.")
    
    # 4. 关联 Echo DICOM 图像路径
    matched_full = pd.merge(matched_echo, df_record[['study_id', 'echo_dicom_path_relative']], on='study_id', how='left')
    matched_full = pd.merge(matched_full, df_cvd_keys[['subject_id', 'hadm_id']].drop_duplicates(), on=['subject_id', 'hadm_id'], how='left')
    matched_full.drop(columns=['admittime', 'dischtime'], inplace=True) 
    
    # 统计信息 (加入统计信息)
    num_matched_dicoms = matched_full['echo_dicom_path_relative'].nunique()
    logger.info(f"Total unique Echo DICOM paths matched: {num_matched_dicoms}.")

    # ... (保存 Echo Details 文件逻辑不变) ...
    echo_output_cols = ['subject_id', 'hadm_id', 'study_id', 'echo_time', 'note_id', 'note_charttime', 'echo_dicom_path_relative']
    df_echo_output = matched_full.copy()
    
    note_content_path = os.path.join(OUTPUT_DIR, f'cvd_matched_notes_content_{suffix}.csv.gz') 
    if os.path.exists(note_content_path):
        df_note_content = pd.read_csv(note_content_path, compression='gzip', 
                                      usecols=['note_id', 'text']).rename(columns={'text': 'echo_note_text'})
        df_note_content = df_note_content.drop_duplicates(subset=['note_id']) 
        df_echo_output = pd.merge(df_echo_output, df_note_content, on='note_id', how='left')
        if 'echo_note_text' in df_echo_output.columns and 'echo_note_text' not in echo_output_cols:
            echo_output_cols.append('echo_note_text') 
        
    output_details_filename = f'cvd_matched_echo_details_{suffix}.csv.gz' 
    df_echo_output[echo_output_cols].to_csv(os.path.join(OUTPUT_DIR, output_details_filename), index=False, compression='gzip')
    logger.info(f"Saved matched Echo Details (Study/Path/Note): {output_details_filename}")


    # 5. 聚合 Echo 详细信息到列表 (HADm 级别)
    # ... (聚合和合并逻辑不变) ...
    echo_agg_cols = {
        'study_id': ('matched_echo_study_ids', lambda x: list(set(x.dropna().astype('Int64')))), 
        'echo_time': ('matched_echo_study_times', lambda x: list(set(x.dropna().astype(str)))), 
        'note_id': ('matched_echo_mimic_note_ids', lambda x: list(set(x.dropna().astype(str)))), 
        'note_charttime': ('matched_echo_note_charttimes', lambda x: list(set(x.dropna().astype(str)))), 
        'echo_dicom_path_relative': ('matched_echo_dicom_paths', lambda x: list(x.dropna().astype(str)))
    }

    agg_dict = {col_in: agg_func for col_in, (col_out, agg_func) in echo_agg_cols.items() if col_in in matched_full.columns}
    df_echo_lists = matched_full.groupby('hadm_id').agg(agg_dict).reset_index()
    rename_map = {col_in: col_out for col_in, (col_out, agg_func) in echo_agg_cols.items() if col_in in matched_full.columns}
    df_echo_lists.rename(columns=rename_map, inplace=True)

    df_echo_lists['echo_study_count'] = df_echo_lists['matched_echo_study_ids'].apply(len) if 'matched_echo_study_ids' in df_echo_lists.columns else 0
    df_echo_lists['echo_dicom_count'] = df_echo_lists['matched_echo_dicom_paths'].apply(len) if 'matched_echo_dicom_paths' in df_echo_lists.columns else 0
    df_echo_lists['has_echo'] = 1

    # 6. 最终合并与保存
    df_final = pd.merge(df_cvd_ecg, df_echo_lists, on='hadm_id', how='left')
    df_final['has_echo'] = df_final['has_echo'].fillna(0).astype(int)
    df_final['echo_study_count'] = df_final['echo_study_count'].fillna(0).astype(int)
    df_final['echo_dicom_count'] = df_final['echo_dicom_count'].fillna(0).astype(int)
    
    list_cols = [c for c in df_echo_lists.columns if c not in ['hadm_id', 'has_echo', 'echo_study_count', 'echo_dicom_count']]
    for col in list_cols:
        df_final[col] = df_final[col].apply(lambda x: x if isinstance(x, list) else [])
    
    output_details_file = f'step_2d_details_cvd_with_all_mm_{suffix}.csv.gz'
    df_final.to_csv(os.path.join(OUTPUT_DIR, output_details_file), index=False, compression='gzip')

    logger.info("Step 2D: Echo Pipeline completed successfully.")
    return df_final

# ================= 3. Clean 版本创建阶段 (Step 3: Clean Version) =================

def step_3_create_clean_version(run_mode, df_final_input):
    suffix = run_mode.lower()
    OUTPUT_DIR_BASE = PATHS['step2_output_dir']
    OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, 'step_3_clean_version') 
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    logger = setup_logger(OUTPUT_DIR_BASE, f'step_3_create_clean_{suffix}.log', 'Step3_Clean_Processor')
    logger.info("="*60)
    logger.info(f"Starting Step 3: Clean Version Creation for [{run_mode}] mode")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    if df_final_input is None or df_final_input.empty:
        logger.error("Input DataFrame for Clean Version is empty. Skipping.")
        return

    df_clean = df_final_input.copy()
    
    cols_to_check = ['has_note', 'has_cxr', 'has_ecg', 'has_echo']
    for col in cols_to_check:
        if col not in df_clean.columns:
            df_clean[col] = 0
        df_clean[col] = df_clean[col].fillna(0).astype(int)

    # 统计信息 (加入统计信息)
    num_input_records = len(df_clean)
    num_input_adm = df_clean['hadm_id'].nunique()
    logger.info(f"Input (Step 2D Output) Records: {num_input_records}, unique admissions: {num_input_adm}.")

    # 核心筛选条件：四列中至少有一个值为 1
    filter_mask = (df_clean['has_note'] == 1) | \
                  (df_clean['has_cxr'] == 1) | \
                  (df_clean['has_ecg'] == 1) | \
                  (df_clean['has_echo'] == 1)
    
    df_filtered = df_clean[filter_mask].copy()

    # 统计信息 (加入统计信息)
    num_clean_records = len(df_filtered)
    num_clean_adm = df_filtered['hadm_id'].nunique()
    num_clean_subjects = df_filtered['subject_id'].nunique()
    
    logger.info(f"--- Clean (Multi-modal Matched) Cohort Statistics ---")
    logger.info(f"Total Input Admissions: {num_input_adm}")
    logger.info(f"Filtered Clean Admissions: {num_clean_adm} (占比: {num_clean_adm/num_input_adm:.2%})" if num_input_adm > 0 else f"Filtered Clean Admissions: {num_clean_adm}")
    logger.info(f"Filtered Clean Records: {num_clean_records}")
    logger.info(f"Filtered Clean Subjects: {num_clean_subjects}")
    logger.info("-------------------------------------------------")


    if df_filtered.empty:
        logger.warning("No records matched any modality. Clean version will be empty.")
        return

    # --- 重点修改: 定义 Clean Labels 和 Clean Details 的列顺序 ---
    
    # ... (列定义和保存逻辑不变) ...
    ID_DIAG_COLS = ['subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version', 'long_title']
    CVD_CAT_COLS = ['CVD_coarse_category', 'CVD_fine_category']
    TARGET_COLS = ['mortality_in_hospital', 'mortality_30d', 'readmission_30d_hosp', 'readmission_30d_icu']
    MM_FLAG_COUNT_COLS = [
        'has_note', 'note_count', 
        'has_cxr', 'cxr_study_count', 'cxr_image_count', 
        'has_ecg', 'ecg_count', 
        'has_echo', 'echo_study_count', 'echo_dicom_count'
    ]
    
    CLEAN_LABELS_COLUMNS = ID_DIAG_COLS + CVD_CAT_COLS + TARGET_COLS + MM_FLAG_COUNT_COLS
    CLEAN_LABELS_COLUMNS = [c for c in CLEAN_LABELS_COLUMNS if c in df_filtered.columns] 
    
    TIME_ANCHOR_COLS = [
        'admittime', 'dischtime',
        'dod', 'final_death_date', 'days_disch_to_death',
        'next_hadm_id', 'next_admittime', 'days_to_next_admission',
        'curr_stay_id', 'next_stay_id'
    ]
    MM_LIST_COLS = [c for c in df_filtered.columns if c.startswith('matched_')]
    
    CLEAN_DETAILS_COLUMNS = CLEAN_LABELS_COLUMNS + TIME_ANCHOR_COLS + MM_LIST_COLS
    CLEAN_DETAILS_COLUMNS = [c for c in CLEAN_DETAILS_COLUMNS if c in df_filtered.columns]


    # 1. 保存 Clean Labels 文件 
    clean_labels_filename = f'step_3_cvd_mmdata_labels_{suffix}.csv.gz'
    clean_labels_path = os.path.join(OUTPUT_DIR, clean_labels_filename)
    
    df_filtered[CLEAN_LABELS_COLUMNS].to_csv(clean_labels_path, index=False, compression='gzip')
    logger.info(f"Saved Clean Labels (Ordered): {clean_labels_path}")

    # 2. 保存 Clean Details 文件
    clean_details_filename = f'step_3_cvd_mmdata_details_{suffix}.csv.gz'
    clean_details_path = os.path.join(OUTPUT_DIR, clean_details_filename)
    
    df_filtered[CLEAN_DETAILS_COLUMNS].to_csv(clean_details_path, index=False, compression='gzip')
    logger.info(f"Saved Clean Details (Ordered): {clean_details_path}")

    logger.info("Step 3: Clean Version Creation completed successfully.")


# ================= 主程序入口 (Main Execution) =================

def main_pipeline(run_mode):
    print(f"--- Starting MIMIC-IV CVD Multi-modal Pipeline in [{run_mode}] Mode ---")
    
    # 0. 检查路径
    if not os.path.exists(PATHS['hosp']) or not os.path.exists(PATHS['icu']):
        print(f"FATAL ERROR: MIMIC-IV base path not found. Check hosp/icu paths: {PATHS['hosp']}")
        return

    # 1. Step 0: 生成死亡/再入院标签
    df_labels_details, success_0 = step_0_generate_labels(run_mode)

    if success_0:
        
        # 2. Step 1: CVD 过滤和标签添加
        df_cvd_cohort, success_1 = step_1_cvd_labeling(run_mode)
        
        if success_1 and not df_cvd_cohort.empty:
        
            # 3. Step 2A-2D: 多模态匹配
            print("\n--- Starting Multi-modal Matching (Step 2) ---")
            # Step 2A: Notes 匹配
            df_note_result = step_2a_note_matching(run_mode, df_cvd_cohort)
            # Step 2B: CXR 匹配
            df_cxr_result = step_2b_cxr_matching(run_mode, df_note_result)
            # Step 2C: ECG 匹配
            df_ecg_result = step_2c_ecg_matching(run_mode, df_cxr_result)
            # Step 2D: Echo 匹配
            df_echo_result = step_2d_echo_matching(run_mode, df_ecg_result)
            
            # 4. Step 3: 创建 Clean 版本
            print("\n--- Starting Clean Version Creation (Step 3) ---")
            step_3_create_clean_version(run_mode, df_echo_result)
            
        elif not success_1:
            print("Step 1 failed. Stopping pipeline.")
        else:
            print("Step 1 returned an empty CVD cohort. Stopping pipeline.")
            
    elif not success_0:
        print("Step 0 failed. Stopping pipeline.")
    else:
        print("Step 0 returned an empty base cohort. Stopping pipeline.")
        
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal data matching pipeline (Notes, CXR, ECG, Echo) for CVD cohort.")
    parser.add_argument('--mode', type=str, default='DEBUG', choices=['DEBUG', 'FULL'],
                        help="Run mode: DEBUG (limited processing) or FULL (all data).")
    args = parser.parse_args()
    
    main_pipeline(args.mode)