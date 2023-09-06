from creditcard.pipeline.batch_prediction import CreditCardBatchPrediction
from creditcard.entity.config_entity import BatchPredictionConfig

if __name__=="__main__":
    batch_prediction_config = BatchPredictionConfig()
    creditcards_batch_prediction = CreditCardBatchPrediction(batch_config = batch_prediction_config)
    creditcards_batch_prediction.start_prediction()