import os
import pandas as pd

# 1) Point this at your CBIS-DDSM mass metadata
BASE = "data/ddsm/"
train_csv = os.path.join(BASE, "mass_case_description_train_set.csv")
test_csv  = os.path.join(BASE, "mass_case_description_test_set.csv")

# 2) Load and concatenate
df_train = pd.read_csv(train_csv)
df_test  = pd.read_csv(test_csv)
#df       = pd.concat([df_train, df_test], ignore_index=True)

df = df_train
# 3) Build full DICOM path
df["dcm_path"] = df["image file path"].apply(lambda p: os.path.join(BASE, p))

# 4) Group by patient_id
pairs = []
for pid, grp in df.groupby("patient_id"):
    paths = grp.sort_values("dcm_path")["dcm_path"].tolist()
    # if they have at least two exams, take first as 'prior', last as 'current'
    if len(paths) >= 2:
        prior, current = paths[0], paths[-1]
        pairs.append((prior, current))

print(f"Found {len(pairs)} longitudinal pairs.")
# pairs is now a list of (P_i, C_i) you can feed into UFCN
