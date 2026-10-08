//+------------------------------------------------------------------+
//|                                              RNN_ONNX.mqh        |
//|   Прямое подключение двух GRU-моделей (BUY/SELL) к советнику     |
//|   через ONNX Runtime (MQL5 ONNX API, сборка 6093).               |
//|                                                                  |
//|   Стратегия загрузки:                                            |
//|   1) OnnxCreateFromBuffer (буфер) - работает и на графике, и в   |
//|      тестере. Требует САМОДОСТАТОЧНЫЙ файл (веса внутри, без     |
//|      .onnx.data) - создаётся скриптом export_onnx.py.            |
//|   2) Fallback OnnxCreate(файл) - работает на графике, в т.ч.     |
//|      для моделей с внешними данными (.onnx.data). В тестере      |
//|      может не сработать (ошибка 5019).                           |
//|                                                                  |
//|   Файлы (в Common\Files\RNN_GOLD\ или MQL5\Files\RNN_GOLD\):     |
//|     buy_model.onnx   sell_model.onnx                             |
//|   Вход:  float32 [1, SEQ_LEN, 23] (форма задаётся переменными    |
//|          g_nn_seq_len / NN_FEATURES в блоке "ПАРАМЕТРЫ МОДЕЛИ")  |
//|   Выход: float32 [1, 2]       (логиты классов)                  |
//|                                                                  |
//|   Признаки строятся ТОЧНО как в feature_pipeline.py (RNN_gold):  |
//|   23 фичи с ATR-релятивизацией от свежайшего бара окна +         |
//|   канальная z-нормализация (ddof=0) по маске за 20 баров.        |
//+------------------------------------------------------------------+
#ifndef RNN_ONNX_MQH
#define RNN_ONNX_MQH

#include "RNN_Scaler.mqh"

// Класс "прибыльной" сделки в выходе модели.
// feature_pipeline.py/PrepareData: label=1 => прибыль; one-hot => класс 1.
#define NN_PROFIT_CLASS   1

//+------------------------------------------------------------------+
//| ПАРАМЕТРЫ МОДЕЛИ берутся из RNN_Scaler.mqh (генерируется         |
//| feature_pipeline.py): NN_BUY_NORM_WINDOW, NN_SELL_NORM_WINDOW,   |
//| NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN, NN_FEATURES, NN_NORM_STD_EPS.   |
//+------------------------------------------------------------------+
#define NN_NUM_CLASSES    2
int g_nn_seq_len[2]    = {NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN};
int g_nn_input_size[2] = {NN_BUY_SEQ_LEN * NN_FEATURES, NN_SELL_SEQ_LEN * NN_FEATURES};

//--- Хендлы ONNX-сессий (0 = BUY, 1 = SELL) ---
long   g_onnx_handle[2] = {INVALID_HANDLE, INVALID_HANDLE};
string g_onnx_file[2];            // имена загруженных ONNX-файлов (для лога)
bool   g_onnx_ready     = false;

//--- Формы тензоров (заполняются в NN_ONNX_ApplyConfig) ---
long   g_nn_input_shape_buy[3];
long   g_nn_input_shape_sell[3];
long   g_nn_output_shape[2] = {1, NN_NUM_CLASSES};

//--- Буферы для inference (модель float32 -> данные float32) ---
float  g_nn_input_buy[];
float  g_nn_input_sell[];
float  g_nn_output[NN_NUM_CLASSES];
double g_nn_raw_buy[];
double g_nn_raw_sell[];

//--- Каналы канонической z-нормализации (NORMALIZE_MASK) ---
//   Из feature_pipeline.py: (0,1,2,3,4,5,6,7,9,10,11,12,13,14,15).
//   Не нормализуются: 8 (close_position), 16-22 (rsi/stoch/время).
int g_nn_norm_mask[15] = {0,1,2,3,4,5,6,7,9,10,11,12,13,14,15};
bool g_nn_norm_flag[NN_FEATURES];

//+------------------------------------------------------------------+
//| Флаг нормализации по каналу (заполняется один раз)               |
//+------------------------------------------------------------------+
void NN_BuildNormMask()
  {
   ArrayInitialize(g_nn_norm_flag, false);
   for(int i = 0; i < ArraySize(g_nn_norm_mask); i++)
      g_nn_norm_flag[g_nn_norm_mask[i]] = true;
  }

//+------------------------------------------------------------------+
//| Применение параметров конфига (RNN_Scaler.mqh) к буферам/формам  |
//+------------------------------------------------------------------+
bool NN_ONNX_ApplyConfig()
  {
   NN_BuildNormMask();

   int seq[2] = {NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN};
   int window[2] = {NN_BUY_NORM_WINDOW, NN_SELL_NORM_WINDOW};
   for(int d = 0; d < 2; d++)
     {
      if(seq[d] < 1 || seq[d] > window[d])
        {
         PrintFormat("ONNX[%s]: seq_len=%d несовместим с norm_window=%d",
                     d == 0 ? "BUY" : "SELL", seq[d], window[d]);
         return(false);
        }
      g_nn_seq_len[d] = seq[d];
      g_nn_input_size[d] = seq[d] * NN_FEATURES;
     }
   if(!ArrayResize(g_nn_raw_buy, NN_BUY_NORM_WINDOW * NN_FEATURES) ||
      !ArrayResize(g_nn_raw_sell, NN_SELL_NORM_WINDOW * NN_FEATURES) ||
      !ArrayResize(g_nn_input_buy, g_nn_input_size[0]) ||
      !ArrayResize(g_nn_input_sell, g_nn_input_size[1]))
      return(false);
   g_nn_input_shape_buy[0] = 1; g_nn_input_shape_buy[1] = seq[0]; g_nn_input_shape_buy[2] = NN_FEATURES;
   g_nn_input_shape_sell[0] = 1; g_nn_input_shape_sell[1] = seq[1]; g_nn_input_shape_sell[2] = NN_FEATURES;
   PrintFormat("ONNX: BUY norm=%d seq=%d, SELL norm=%d seq=%d, features=%d",
               window[0], seq[0], window[1], seq[1], NN_FEATURES);
   return(true);
  }

//+------------------------------------------------------------------+
//| Чтение файла модели в буфер и создание сессии из буфера         |
//| (работает и на графике, и в тестере стратегий)                  |
//+------------------------------------------------------------------+
long NN_OnnxLoadFromBuffer(const string model_path, const int direction)
  {
   string tag = (direction == 0) ? "BUY" : "SELL";

   ResetLastError();
   int file = FileOpen(model_path, FILE_READ | FILE_BIN);
   if(file == INVALID_HANDLE)
      file = FileOpen(model_path, FILE_READ | FILE_BIN | FILE_COMMON);
   if(file == INVALID_HANDLE)
     {
      PrintFormat("ONNX[%s]: не удалось открыть файл '%s' (код %d)",
                  tag, model_path, GetLastError());
      return(INVALID_HANDLE);
     }

   ulong size = FileSize(file);
   if(size <= 0)
     {
      PrintFormat("ONNX[%s]: файл '%s' пуст", tag, model_path);
      FileClose(file);
      return(INVALID_HANDLE);
     }

   uchar buffer[];
   if(!ArrayResize(buffer, (int)size))
     {
      PrintFormat("ONNX[%s]: недостаточно памяти для '%s' (%I64u байт)",
                  tag, model_path, size);
      FileClose(file);
      return(INVALID_HANDLE);
     }

   if(FileReadArray(file, buffer, 0, (int)size) != (int)size)
     {
      PrintFormat("ONNX[%s]: ошибка чтения '%s' (код %d)",
                  tag, model_path, GetLastError());
      FileClose(file);
      return(INVALID_HANDLE);
     }
   FileClose(file);

   long h = OnnxCreateFromBuffer(buffer, 0);
   if(h == INVALID_HANDLE)
     {
      PrintFormat("ONNX[%s]: OnnxCreateFromBuffer('%s') не удался (код %d), пробуем OnnxCreate...",
                  tag, model_path, GetLastError());
      ResetLastError();
      h = OnnxCreate(model_path, 0);
      if(h == INVALID_HANDLE)
        {
         PrintFormat("ONNX[%s]: OnnxCreate('%s') тоже не удался (код %d).",
                     tag, model_path, GetLastError());
         return(INVALID_HANDLE);
        }
      PrintFormat("ONNX[%s]: загружена через OnnxCreate(файл) '%s'", tag, model_path);
     }

   //--- Явно задаём формы входов/выходов ---
   bool shape_ok = false;
   if(direction == 0)
      shape_ok = OnnxSetInputShape(h, 0, g_nn_input_shape_buy);
   else
      shape_ok = OnnxSetInputShape(h, 0, g_nn_input_shape_sell);
   if(!shape_ok)
      PrintFormat("ONNX[%s]: OnnxSetInputShape (код %d)", tag, GetLastError());
   if(!OnnxSetOutputShape(h, 0, g_nn_output_shape))
      PrintFormat("ONNX[%s]: OnnxSetOutputShape (код %d)", tag, GetLastError());

   //--- Диагностика ---
   long in_count  = OnnxGetInputCount(h);
   long out_count = OnnxGetOutputCount(h);
   string in_name  = (in_count > 0)  ? OnnxGetInputName(h, 0)  : "?";
   string out_name = (out_count > 0) ? OnnxGetOutputName(h, 0) : "?";

   string our_shape = "";
   long shape_value[3];
   shape_value[0] = 1;
   shape_value[1] = g_nn_seq_len[direction];
   shape_value[2] = NN_FEATURES;
   for(int i = 0; i < 3; i++)
      our_shape += (i > 0 ? "x" : "") + IntegerToString(shape_value[i]);

   PrintFormat("ONNX[%s]: '%s' готова (%I64u байт) | вход: '%s' [%s], выход: '%s' 2",
               tag, model_path, size, in_name, our_shape, out_name);

   g_onnx_handle[direction] = h;
   g_onnx_file[direction]   = model_path;
   return(h);
  }

//+------------------------------------------------------------------+
//| Инициализация: загрузка обеих моделей                            |
//+------------------------------------------------------------------+
bool NN_ONNX_Init(const string buy_model_file, const string sell_model_file)
  {
   g_onnx_ready = false;

   if(!NN_ONNX_ApplyConfig())
      return(false);

   if(NN_OnnxLoadFromBuffer(buy_model_file, 0) == INVALID_HANDLE)
      return(false);
   if(NN_OnnxLoadFromBuffer(sell_model_file, 1) == INVALID_HANDLE)
     {
      NN_ONNX_Deinit();
      return(false);
     }

   g_onnx_ready = true;
   return(true);
  }

//+------------------------------------------------------------------+
//| Деинициализация: освобождение сессий                             |
//+------------------------------------------------------------------+
void NN_ONNX_Deinit()
  {
   for(int i = 0; i < 2; i++)
     {
      if(g_onnx_handle[i] != INVALID_HANDLE)
        {
         OnnxRelease(g_onnx_handle[i]);
         g_onnx_handle[i] = INVALID_HANDLE;
        }
     }
   g_onnx_ready = false;
  }

//+------------------------------------------------------------------+
//| Клип в [0,1] (аналог np.clip)                                    |
//+------------------------------------------------------------------+
double NN_Clamp01(const double v)
  {
   if(v < 0.0) return(0.0);
   if(v > 1.0) return(1.0);
   return(v);
  }

//+------------------------------------------------------------------+
//| Канонический признак одного бара (23 фичи, контракт 20x23).       |
//| Релятивизация от самого свежего бара окна (lag0_close/lag0_atr), |
//| см. feature_pipeline.py / derive_features.                       |
//+------------------------------------------------------------------+
void NN_FillBarFeatures(const int bar,
                        const double &open[],
                        const double &close[],
                        const double &high[],
                        const double &low[],
                        const long   &volume[],
                        const double &ema8[],
                        const double &ema21[],
                        const double &rsi[],
                        const double &stochK[],
                        const double &stochD[],
                        const double &macd_main[],
                        const double &macd_signal[],
                        const double &atr[],
                        const double lag0_close,
                        const double lag0_atr,
                        double &out[])
  {
   double o = open[bar];
   double c = close[bar];
   double h = high[bar];
   double l = low[bar];
   double atr0 = MathMax(lag0_atr, NN_NORM_STD_EPS);
   double atr_e  = MathMax(atr[bar], NN_NORM_STD_EPS);
   double range  = MathMax(h - l, NN_NORM_STD_EPS);

   out[0]  = (o - lag0_close) / atr0;                       // open_rel_atr
   out[1]  = (h - lag0_close) / atr0;                       // high_rel_atr
   out[2]  = (l - lag0_close) / atr0;                       // low_rel_atr
   out[3]  = (c - lag0_close) / atr0;                       // close_rel_atr
   out[4]  = (c - o) / atr_e;                               // body_atr
   out[5]  = (h - l) / atr_e;                               // range_atr
   out[6]  = (h - MathMax(o, c)) / atr_e;                   // upper_wick_atr
   out[7]  = (MathMin(o, c) - l) / atr_e;                   // lower_wick_atr
   out[8]  = NN_Clamp01((c - l) / range);                   // close_position
   out[9]  = MathLog(1.0 + MathMax((double)volume[bar], 0.0)); // log_volume_relative (raw)
   out[10] = (ema8[bar] - c) / atr_e;                       // ema8_rel_atr
   out[11] = (ema21[bar] - c) / atr_e;                      // ema21_rel_atr
   out[12] = (ema8[bar] - ema21[bar]) / atr_e;              // ema_spread_atr
   out[13] = macd_main[bar] / atr_e;                        // macd_main_atr
   out[14] = (macd_main[bar] - macd_signal[bar]) / atr_e;   // macd_hist_atr
   out[15] = atr[bar] / atr0;                               // atr_relative
   out[16] = NN_Clamp01(rsi[bar] / 100.0);                  // rsi_01
   out[17] = NN_Clamp01(stochK[bar] / 100.0);               // stoch_k_01
   out[18] = NN_Clamp01(stochD[bar] / 100.0);               // stoch_d_01

   datetime bt = iTime(_Symbol, PERIOD_M15, bar);
   MqlDateTime dt;
   TimeToStruct(bt, dt);
   double hour = dt.hour + dt.min / 60.0;
   out[19] = MathSin(2.0 * M_PI * hour / 24.0);             // sin_hour
   out[20] = MathCos(2.0 * M_PI * hour / 24.0);             // cos_hour
   double dow = (double)dt.day_of_week / 6.0;               // day_of_week (0..1)
   out[21] = MathSin(2.0 * M_PI * dow / 7.0);               // sin_day_of_week
   out[22] = MathCos(2.0 * M_PI * dow / 7.0);               // cos_day_of_week
  }

//+------------------------------------------------------------------+
//| Построение вектора СЫРЫХ признаков (norm_window баров x 23).     |
//| Порядок баров: сигнальный (самый свежий, lag0) -> более старые. |
//| Если истории меньше norm_window — недостающие (старые) бары      |
//| добиваются средним по доступным барам.                           |
//+------------------------------------------------------------------+
void NN_BuildFeatures(const int signal_shift,
                      const double &open[],
                      const double &close[],
                      const double &high[],
                      const double &low[],
                      const long   &volume[],
                      const double &ema8[],
                      const double &ema21[],
                      const double &rsi[],
                      const double &stochK[],
                      const double &stochD[],
                      const double &macd_main[],
                      const double &macd_signal[],
                      const double &atr[],
                      const int norm_window,
                      const int avail_bars,
                      double &raw[])
  {
   double bar_feat[NN_FEATURES];
   double feat_mean[NN_FEATURES];
   ArrayInitialize(feat_mean, 0.0);

   // lag0 = сигнальный (самый свежий) бар окна
   double lag0_close = close[signal_shift];
   double lag0_atr   = atr[signal_shift];

   int n_real = avail_bars - signal_shift;
   if(n_real > norm_window)
      n_real = norm_window;
   if(n_real < 1)
      n_real = 1;

   // --- проход 1: средние по реальным барам (для добивки старых) ---
   for(int k = 0; k < n_real; k++)
     {
      NN_FillBarFeatures(signal_shift + k, open, close, high, low, volume,
                         ema8, ema21, rsi, stochK, stochD,
                         macd_main, macd_signal, atr,
                         lag0_close, lag0_atr, bar_feat);
      for(int f = 0; f < NN_FEATURES; f++)
         feat_mean[f] += bar_feat[f];
     }
   for(int f = 0; f < NN_FEATURES; f++)
      feat_mean[f] /= (double)n_real;

   // --- проход 2: заполнение полного окна (реальные бары + среднее) ---
   int idx = 0;
   for(int k = 0; k < norm_window; k++)
     {
      if(k < n_real)
        {
         NN_FillBarFeatures(signal_shift + k, open, close, high, low, volume,
                            ema8, ema21, rsi, stochK, stochD,
                            macd_main, macd_signal, atr,
                            lag0_close, lag0_atr, bar_feat);
         for(int f = 0; f < NN_FEATURES; f++)
            raw[idx++] = bar_feat[f];
        }
      else
        {
         for(int f = 0; f < NN_FEATURES; f++)
            raw[idx++] = feat_mean[f];
        }
     }
  }

//+------------------------------------------------------------------+
//| Медиана массива (копия сортируется)                              |
//+------------------------------------------------------------------+
double NN_Median(const double &src[], const int n)
  {
   double a[];
   ArrayResize(a, n);
   ArrayCopy(a, src, 0, 0, n);
   ArraySort(a);
   if((n & 1) == 1)
      return(a[n / 2]);
   return((a[n / 2 - 1] + a[n / 2]) / 2.0);
  }

//+------------------------------------------------------------------+
//| Каноническая нормализация:                                     |
//|  1) медианное центрирование канала 9 (log_volume_relative) по окну|
//|  2) z-нормализация (ddof=0) каналов из маски {0-7,9-15} по окну |
//|  3) первые seq_len баров -> g_nn_input (float32)                 |
//+------------------------------------------------------------------+
void NN_ApplyRollingNorm(const int direction, const double &raw[])
  {
   double mean[NN_FEATURES];
   double std[NN_FEATURES];
   int norm_window = direction == 0 ? NN_BUY_NORM_WINDOW : NN_SELL_NORM_WINDOW;

   // --- медиана канала 9 по окну ---
   double v9[];
   ArrayResize(v9, norm_window);
   for(int k = 0; k < norm_window; k++)
      v9[k] = raw[k * NN_FEATURES + 9];
   double med9 = NN_Median(v9, norm_window);

   // --- mean/std только для каналов из маски ---
   ArrayInitialize(mean, 0.0);
   ArrayInitialize(std, 0.0);
   for(int f = 0; f < NN_FEATURES; f++)
     {
      if(!g_nn_norm_flag[f])
         continue;

      double s = 0.0;
      for(int k = 0; k < norm_window; k++)
        {
         double v = (f == 9) ? (raw[k * NN_FEATURES + 9] - med9) : raw[k * NN_FEATURES + f];
         s += v;
        }
      mean[f] = s / (double)norm_window;

      double s2 = 0.0;
      for(int k = 0; k < norm_window; k++)
        {
         double v = (f == 9) ? (raw[k * NN_FEATURES + 9] - med9) : raw[k * NN_FEATURES + f];
         double d = v - mean[f];
         s2 += d * d;
        }
      std[f] = MathSqrt(s2 / (double)norm_window);   // ddof=0
      if(std[f] < NN_NORM_STD_EPS)
         std[f] = NN_NORM_STD_EPS;
     }

   // --- заполнение входного тензора первыми seq_len барами окна ---
   for(int k = 0; k < g_nn_seq_len[direction]; k++)
      for(int f = 0; f < NN_FEATURES; f++)
        {
         int i = k * NN_FEATURES + f;
         double base = (f == 9) ? (raw[i] - med9) : raw[i];
         double value = g_nn_norm_flag[f] ? ((base - mean[f]) / std[f]) : base;
         float fv = (float)value;
         if(direction == 0)
            g_nn_input_buy[i] = fv;
         else
            g_nn_input_sell[i] = fv;
        }
  }

//+------------------------------------------------------------------+
//| Прогон одной модели через ONNX Runtime                           |
//| direction: 0 = BUY, 1 = SELL                                     |
//| Возвращает softmax-вероятности классов 0 и 1.                    |
//+------------------------------------------------------------------+
bool NN_OnnxRunModel(const int direction, const double &features[],
                     double &prob_class0, double &prob_class1)
  {
   prob_class0 = 0.0;
   prob_class1 = 0.0;

   if(direction < 0 || direction > 1)
      return(false);
   if(g_onnx_handle[direction] == INVALID_HANDLE)
      return(false);
   int norm_window = direction == 0 ? NN_BUY_NORM_WINDOW : NN_SELL_NORM_WINDOW;
   if(g_nn_input_size[direction] <= 0 || ArraySize(features) < norm_window * NN_FEATURES)
     {
      PrintFormat("ONNX[%s]: размер вектора признаков %d < %d (окно нормализации не настроено?)",
                  (direction == 0) ? "BUY" : "SELL",
                  ArraySize(features), norm_window * NN_FEATURES);
      return(false);
     }

   //--- Каноническая нормализация (маска + медиана канала 9) ---
   NN_ApplyRollingNorm(direction, features);

   //--- Инференс (модель float32, данные float32 -> без конвертации) ---
   ResetLastError();
   bool run_ok = false;
   if(direction == 0)
      run_ok = OnnxRun(g_onnx_handle[direction], 0, g_nn_input_buy, g_nn_output);
   else
      run_ok = OnnxRun(g_onnx_handle[direction], 0, g_nn_input_sell, g_nn_output);
   if(!run_ok)
     {
      PrintFormat("ONNX[%s]: ошибка OnnxRun (код %d)",
                  (direction == 0) ? "BUY" : "SELL", GetLastError());
      return(false);
     }

   //--- Softmax ---
   double m = MathMax((double)g_nn_output[0], (double)g_nn_output[1]);
   double e0 = MathExp((double)g_nn_output[0] - m);
   double e1 = MathExp((double)g_nn_output[1] - m);
   double s  = e0 + e1;
   prob_class0 = e0 / s;
   prob_class1 = e1 / s;
   return(true);
  }

//+------------------------------------------------------------------+
//| Уверенность модели в "прибыльной" сделке для направления.        |
//| direction: 0 = BUY, 1 = SELL. profit_class: класс "прибыли"      |
//| или -1.0 при ошибке.                                             |
//+------------------------------------------------------------------+
double NN_ONNX_Predict(const int direction, const double &features[], const int profit_class = NN_PROFIT_CLASS)
  {
   double p0 = 0.0, p1 = 0.0;
   if(!NN_OnnxRunModel(direction, features, p0, p1))
      return(-1.0);
   double conf = (profit_class == 0) ? p0 : p1;
   PrintFormat("ONNX[%s]: модель '%s' | confidence = %.6f (P0=%.6f P1=%.6f, profit_class=%d)",
               (direction == 0) ? "BUY" : "SELL", g_onnx_file[direction], conf, p0, p1, profit_class);
   return(conf);
  }

//+------------------------------------------------------------------+
//| Готовность модуля                                                |
//+------------------------------------------------------------------+
bool NN_ONNX_IsReady()
  {
   return(g_onnx_ready);
  }

//+------------------------------------------------------------------+
//| Длина окна НС (баров) — из конфига RNN_Scaler.mqh (NN_SEQ_LEN). |
//+------------------------------------------------------------------+
int NN_ONNX_GetSeqLen(const int direction = 0)
  {
   return((direction == 0) ? NN_BUY_SEQ_LEN : NN_SELL_SEQ_LEN);
  }


#endif // RNN_ONNX_MQH