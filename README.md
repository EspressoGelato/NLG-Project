# NLG-Project

1. To construct the PAIR data from Fact2Story raw dataset, one can run:

./pair-emnlp2020/planning/data/compute_topic_signatures.py
./pair-emnlp2020/planning/data/build_facts_data.py

2. To convert Fact2Story template data to PAIR: 

./pair-emnlp2020/planning/data/build_facts_data2.py

3. To train the refinement module on Fact data:
sh ./pair-emnlp2020/refinement/train_fact-data.sh
./pair-emnlp2020/compute_metric.py

4. run Facts2Story-XLNetPlanCloze/convert_id_to_name_prediction.py to convert the prediction

5. To evaluate GRUEN, please refer its instruction.

