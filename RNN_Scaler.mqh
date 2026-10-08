//+------------------------------------------------------------------+
//|                                              RNN_Scaler.mqh      |
//|   Pre-normalized NPZ input (RNN_gold).                         |
//|   Сгенерировано: Colab.                                        |
//+------------------------------------------------------------------+
#ifndef RNN_SCALER_MQH
#define RNN_SCALER_MQH

#define NN_BUY_SEQ_LEN      9       // BUY input window
#define NN_SELL_SEQ_LEN     11       // SELL input window
#define NN_MAX_SEQ_LEN      11
#define NN_FEATURES      23    // признаков на бар
#define NN_NORM_STD_EPS  1e-06         // epsilon

/* Direction-neutral aliases for existing data preparation tools. */
/* Окно нормализации = размерам тренировочного окна (N_BARS = 20):
   z-нормализация/медиана считаются по всем барам окна, в сеть идут
   первые NN_*_SEQ_LEN баров. */
#define NN_BUY_NORM_WINDOW      20
#define NN_SELL_NORM_WINDOW     20
#define NN_MAX_NORM_WINDOW      20

#define NN_SEQ_LEN     NN_BUY_SEQ_LEN

#endif // RNN_SCALER_MQH
