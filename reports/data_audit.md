# CrisisMMD v2.0 Data Audit

## Provenance

- Download manifest: `data/raw/download_manifest.json`
- Source record: `{"source": "official", "artifacts": [{"filename": "CrisisMMD_v2.0.tar.gz", "path": "/data2/yangyz/comp664/comp664-project5/data/raw/archives/CrisisMMD_v2.0.tar.gz", "url": "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz", "bytes": 1902053684, "sha256": "351b9be6123a5d5074407d792bfdddc1f470d4909f65ddd429b584d24607d43f", "retrieved_at": "2026-10-25T20:11:41.395507+00:00"}, {"filename": "crisismmd_datasplit_agreed_label.zip", "path": "/data2/yangyz/comp664/comp664-project5/data/raw/splits/crisismmd_datasplit_agreed_label.zip", "url": "https://crisisnlp.qcri.org/data/crisismmd/crisismmd_datasplit_agreed_label.zip", "bytes": 1598517, "sha256": "32f2203bbf0545057be2c6d5c95636cfb5032aef58d0f57e04c5dfac246843ff", "retrieved_at": "2026-10-25T20:12:04.342002+00:00"}]}`

## Canonical data

### id_official

- Rows: 12708
- Unique tweets: 11400
- Events: california_wildfires, hurricane_harvey, hurricane_irma, hurricane_maria, iraq_iran_earthquake, mexico_earthquake, srilanka_floods
- Tweet overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image ID overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image checksum overlap: `{'train_val': 26, 'train_test': 25, 'val_test': 3}`

### id_official_strict

- Rows: 12639
- Unique tweets: 11334
- Events: california_wildfires, hurricane_harvey, hurricane_irma, hurricane_maria, iraq_iran_earthquake, mexico_earthquake, srilanka_floods
- Tweet overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image ID overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image checksum overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

### ood_heldout

- Rows: 12708
- Unique tweets: 11400
- Events: california_wildfires, hurricane_harvey, hurricane_irma, hurricane_maria, iraq_iran_earthquake, mexico_earthquake, srilanka_floods
- Tweet overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image ID overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image checksum overlap: `{'train_val': 23, 'train_test': 1, 'val_test': 1}`

### ood_heldout_strict

- Rows: 12668
- Unique tweets: 11365
- Events: california_wildfires, hurricane_harvey, hurricane_irma, hurricane_maria, iraq_iran_earthquake, mexico_earthquake, srilanka_floods
- Tweet overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image ID overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`
- Image checksum overlap: `{'train_val': 0, 'train_test': 0, 'val_test': 0}`

## Image validation

- Checked: 12708
- Readable: 12708
- Exceptions: 0 (see `results/image_exceptions.csv`)
- Formats: `{'JPEG': 12617, 'PNG': 91}`

