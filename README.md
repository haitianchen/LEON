# LEON: A New Framework for ML-Aided Query Optimization

Pytorch implementation of LEON: A New Framework for ML-Aided Query Optimization.

##  Requirment  

### Python Environment

```
python 3.8.5
pandas
torch==1.4.0
psycopg2-binary==2.8.5
numpy==1.18.1
networkx
ipdb
Pillow==9.2.0
scikit-learn==1.0.2
scipy==1.7.3
sqlparse
re
```

Run `pip install -r requirements.txt`  to quickly install Python Environment.

### PostgreSQL 

Postgres v12.5

pg_hint_plan v1.3.7

*After installing PostgreSQL and its extension, you need to modify its default configuration that can find in [postgresql.conf](./postgresql.conf)*

### BenchMark

In our paper,we use two benchmark, JOB and TPC-H, you can get it through the following link.

**Join-order-benchmark:** 	https://github.com/gregrahn/join-order-benchmark

**TPC-H:**  https://github.com/electrum/tpch-dbgen

## Usage

First, you need to modify the log_path, model_path and other parameters in the training code(such as [train_job.py](./train_job.py)), and modify the relevant information required to connect to PostgreSQL in [pg_executor.py](./util/pg_executor.py).



## Contact



## Reference
