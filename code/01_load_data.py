"""Load Epoch AI Benchmarking Hub CSVs into one tidy long-format dataset.
Usage: python 01_load_data.py <path_to_epoch_csv_dir> <output_csv>
Keeps only benchmarks whose primary metric is a bounded proportion; normalizes to [0,1].
"""
import pandas as pd, os, sys

# Primary score column per file; None = excluded (unbounded/relative metric)
SCORE_COLS = {
 'adversarial_nli_external.csv':'Score','aider_polyglot_external.csv':'Percent correct',
 'ale_bench_external.csv':None,'algotune_external.csv':None,
 'apex_agents_external.csv':'Pass@1 score','arc_agi_2_external.csv':'Score','arc_agi_external.csv':'Score',
 'arc_ai2_external.csv':'Challenge score','balrog_external.csv':'Average progress','bbh_external.csv':'Average',
 'bool_q_external.csv':'Score','cad_eval_external.csv':'Overall pass (%)','chess_puzzles.csv':'mean_score',
 'cl_bench_external.csv':'Overall','cl_bench_life_external.csv':'Overall','common_sense_qa_2_external.csv':'Score',
 'critpt_external.csv':'Accuracy','cybench_external.csv':'Unguided % Solved','deepresearchbench_external.csv':None,
 'exploitbench_external.csv':'Mean capability','fictionlivebench_external.csv':'120k token score',
 'forecastbench_external.csv':None,'frontiercode_external.csv':'Diamond score','frontiermath.csv':'mean_score',
 'frontiermath_tier_4.csv':'mean_score','frontierswe_external.csv':None,'gbaeval_external.csv':'Overall score',
 'gdp_pdf_external.csv':'GDP.pdf score','gdpval_external.csv':'Win Rate (%)','geobench_external.csv':None,
 'gpqa_diamond.csv':'mean_score','gsm8k_external.csv':'EM','gso_external.csv':'Score OPT@1',
 'hella_swag_external.csv':'Overall accuracy','hle_external.csv':'Accuracy','lambada_external.csv':'Score',
 'lech_mazur_writing_external.csv':None,'live_bench_external.csv':'Global average','math_level_5.csv':'mean_score',
 'metr_time_horizons_external.csv':None,'mmlu_external.csv':'EM','open_book_qa_external.csv':'Accuracy',
 'os_world_external.csv':'Score','osworld_2_external.csv':'Binary accuracy','otis_mock_aime_2024_2025.csv':'mean_score',
 'piqa_external.csv':'Score','posttrainbench_external.csv':'Average (%)','rli_external.csv':'Score',
 'scicode_external.csv':'Score','science_qa_external.csv':'Score','simplebench_external.csv':'Score (AVG@5)',
 'simpleqa_verified.csv':'mean_score','superglue_external.csv':'Score','swe_bench_verified.csv':'mean_score',
 'terminalbench_external.csv':'Accuracy mean','the_agent_company_external.csv':'% Score',
 'trivia_qa_external.csv':'EM','vending_bench_2_external.csv':None,'video_mme_external.csv':'Overall (no subtitles)',
 'vpct_external.csv':'Correct','webdev_arena_external.csv':None,'weirdml_external.csv':'Accuracy',
 'wino_grande_external.csv':'Accuracy','epoch_capabilities_index.csv':None,
}

def main(src, out):
    frames=[]
    for f,col in SCORE_COLS.items():
        p=os.path.join(src,f)
        if col is None or not os.path.exists(p): continue
        df=pd.read_csv(p)
        if col not in df.columns or 'Release date' not in df.columns: continue
        d=df[['Model version','Release date',col]].copy()
        d.columns=['model','date','score']
        d['score']=pd.to_numeric(d['score'],errors='coerce')
        d['date']=pd.to_datetime(d['date'],errors='coerce')
        d=d.dropna()
        if len(d)==0: continue
        if d['score'].max()>1.5: d['score']/=100.0   # percent -> fraction
        d['benchmark']=f.replace('_external.csv','').replace('.csv','')
        frames.append(d)
    all_df=pd.concat(frames)
    all_df.to_csv(out,index=False)
    print(f"{all_df['benchmark'].nunique()} benchmarks, {len(all_df)} rows -> {out}")

if __name__=='__main__':
    main(sys.argv[1], sys.argv[2])
