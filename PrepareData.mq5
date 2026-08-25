//+------------------------------------------------------------------+
//|                                                  PrepareData.mq5 |
//|                                      Copyright 2026, Bondarev A. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Bondarev A."
#property link      "https://www.mql5.com"
#property version   "1.83"
//+------------------------------------------------------------------+
//|                    Экспорт сигналов и разметка в CSV              |
//|   имена файлов: data_[BUY|SELL]_[StartDate]_[EndDate].csv        |
//|   сохранение в Common\Files                                      |
//|   StartDate/EndDate = реальный интервал теста (первый/последний   |
//|   бар). Файл создаётся в начале теста с временным именем и        |
//|   переименовывается в OnDeinit после завершения прогона.         |
//|                                                                  |
//|   Колонки CSV: datetime, signal, lag0_*..lagN_*, result, label   |
//|   result - фактический результат сделки (прибыль+своп, валюта    |
//|   депозита), label - разметка: прибыль -> 1, иначе -> 0.         |
//|                                                                  |
//|   Разметка ПО ФАКТУ ИСПОЛНЕНИЯ ОРДЕРА:                          |
//|   - сигнал -> в тестере открывается рыночный ордер с SL/TP;      |
//|   - закрытие: по TP/SL, по Inp_HoldBars свечам ИЛИ до конца дня |
//|     (что наступит раньше); на последнем баре дня позиции        |
//|     закрываются принудительно, без переноса на следующий день;  |
//|   - по фактическому результату сделки:                           |
//|     прибыль -> label=1, убыток/безубыток/не закрылся -> label=0. |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Files\FileTxt.mqh>

#define EXPORT_SEQ_LEN 20
#define EXPORT_FEATURES 18

// --- Входные параметры ---
// Каждая строка CSV содержит окно из 20 баров: от сигнального к более старым.
input double   Inp_LotSize      = 0.01;      // Объём ордера (лоты)
input int      Inp_Magic        = 777001;    // Магик-номер
input double   Inp_TP_Points    = 200.0;     // Тейк-профит (пункты), 0 - без TP
input double   Inp_SL_Points    = 150.0;     // Стоп-лосс (пункты), 0 - без SL
input int      Inp_HoldBars     = 20;        // Макс. горизонт удержания; также закрытие до конца дня
input bool     Inp_ExportBuy    = true;      // Экспортировать BUY-сигналы
input bool     Inp_ExportSell   = true;      // Экспортировать SELL-сигналы

// --- Глобальные переменные ---
CTrade         trade;
int            handles[6];                          // EMA8, EMA21, RSI, Stoch, MACD, ATR
CFileTxt       csvFileBuy, csvFileSell;              // Файлы для BUY и SELL
bool           files_ok = false;
int            rows_written = 0;
int            buy_signals = 0, sell_signals = 0;   // кол-во сигналов (пошло в торговлю)
int            buy_trades = 0, sell_trades = 0;     // кол-во фактически открытых сделок
int            buy_label0 = 0, buy_label1 = 0;      // записано строк BUY: label=0 / label=1
int            sell_label0 = 0, sell_label1 = 0;    // записано строк SELL: label=0 / label=1
double         tp_price = 0.0, sl_price = 0.0;
  int            total_features = 0;              // размер вектора признаков (20 * EXPORT_FEATURES)

// --- Отслеживание открытых сделок ---
struct DealRecord
  {
   ulong         ticket;        // тикет позиции
   int           type;          // 0 - BUY, 1 - SELL
   datetime      signal_time;   // время бара, на котором возник сигнал
   int           held_bars;     // сколько баров позиция уже открыта
   double        features[];    // признаковый вектор
  };
DealRecord     deals[];
int            deals_count = 0;
datetime       last_bar_time = 0;
datetime       test_start = 0;                      // первый бар теста
datetime       test_end = 0;                        // последний бар теста
string         fnameBuy = "";                       // текущее имя BUY-файла (временное)
string         fnameSell = "";                      // текущее имя SELL-файла (временное)

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   if(!MQLInfoInteger(MQL_TESTER))
     {
      Print("Советник предназначен только для тестера стратегий!");
      return(INIT_FAILED);
     }

  if(Inp_HoldBars < 1 || Inp_LotSize <= 0 ||
      Inp_TP_Points < 0.0 || Inp_SL_Points < 0.0)
     {
    Print("Проверьте параметры: Inp_HoldBars>=1, Inp_LotSize>0, TP/SL>=0");
      return(INIT_FAILED);
     }

   // 1 пункт = 1 шаг цены (минимум изменения)

   tp_price = Inp_TP_Points * _Point;
   sl_price = Inp_SL_Points * _Point;

  total_features = EXPORT_SEQ_LEN * EXPORT_FEATURES;

   handles[0] = iMA(_Symbol, PERIOD_M15, 8, 0, MODE_EMA, PRICE_CLOSE);
   handles[1] = iMA(_Symbol, PERIOD_M15, 21, 0, MODE_EMA, PRICE_CLOSE);
   handles[2] = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   handles[3] = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   handles[4] = iMACD(_Symbol, PERIOD_M15, 8, 17, 6, PRICE_CLOSE);
   handles[5] = iATR(_Symbol, PERIOD_M15, 14);

   for(int i=0; i<6; i++)
     {
      if(handles[i] == INVALID_HANDLE)
        {
         PrintFormat("Ошибка создания индикатора %d", i);
         return(INIT_FAILED);
        }
     }

   trade.SetExpertMagicNumber(Inp_Magic);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFillingBySymbol(_Symbol);

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Закрываем оставшиеся позиции и записываем их результат
   for(int i = deals_count - 1; i >= 0; i--)
     {
      int label = 0;
      double result = 0.0;
      if(PositionSelectByTicket(deals[i].ticket))
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         if(trade.PositionClose(deals[i].ticket))
            label = (profit > 0.0) ? 1 : 0;
         result = profit;
        }
      else
        {
         // позиция уже закрыта (TP/SL и т.п.) - результат по истории сделок
         double p = 0.0;
         if(GetClosedDealProfit(deals[i].ticket, p))
            label = (p > 0.0) ? 1 : 0;
         result = p;
        }
      WriteDealRow(deals[i].type, deals[i].signal_time, deals[i].features, result, label);
     }
   deals_count = 0;
   ArrayResize(deals, 0);

   for(int i=0; i<6; i++)
      IndicatorRelease(handles[i]);
   csvFileBuy.Close();
   csvFileSell.Close();

   // Переименовываем файлы: имя должно отражать реальный интервал теста
   test_end = last_bar_time;
   if(test_end == 0)
      test_end = test_start;
   if(files_ok)
     {
      string new_buy = GenerateFileName("BUY", test_start, test_end);
      if(fnameBuy != "" && fnameBuy != new_buy)
        {
         if(FileIsExist(new_buy, FILE_COMMON))
            FileDelete(new_buy, FILE_COMMON);
         FileMove(fnameBuy, FILE_COMMON, new_buy, FILE_COMMON);
        }
      string new_sell = GenerateFileName("SELL", test_start, test_end);
      if(fnameSell != "" && fnameSell != new_sell)
        {
         if(FileIsExist(new_sell, FILE_COMMON))
            FileDelete(new_sell, FILE_COMMON);
         FileMove(fnameSell, FILE_COMMON, new_sell, FILE_COMMON);
        }
     }

  }

//+------------------------------------------------------------------+
//| Expert tick function: обрабатываем только на новой свече         |
//+------------------------------------------------------------------+
void OnTick()
  {
   datetime cur_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(cur_bar == last_bar_time)
      return;
   last_bar_time = cur_bar;

   if(test_start == 0)
      test_start = cur_bar;

   if(!files_ok)
      EnsureFilesOpen();

   // На последнем баре дня закрываем все позиции до конца дня
   if(IsLastBarOfDay(cur_bar))
     {
      CloseAllDeals();
      return;   // новые сигналы в конце дня не открываем
     }

   ProcessOpenDeals();   // закрытие позиций по горизонту / обработка TP/SL
   DetectAndTrade();     // новые сигналы
  }

//+------------------------------------------------------------------+
//| Признак последнего бара торгового дня                            |
//+------------------------------------------------------------------+
bool IsLastBarOfDay(datetime bar_time)
  {
   MqlDateTime dt, dt_next;
   TimeToStruct(bar_time, dt);
   TimeToStruct(bar_time + PeriodSeconds(PERIOD_M15), dt_next);
   return(dt.day != dt_next.day);
  }

//+------------------------------------------------------------------+
//| Принудительное закрытие всех позиций до конца дня                |
//+------------------------------------------------------------------+
void CloseAllDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      int label = 0;
      double result = 0.0;
      if(PositionSelectByTicket(deals[i].ticket))
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         if(trade.PositionClose(deals[i].ticket))
            label = (profit > 0.0) ? 1 : 0;
         result = profit;
        }
      else
        {
         // позиция уже закрыта (TP/SL) - результат по истории сделок
         double p = 0.0;
         if(GetClosedDealProfit(deals[i].ticket, p))
            label = (p > 0.0) ? 1 : 0;
         result = p;
        }
      WriteDealRow(deals[i].type, deals[i].signal_time, deals[i].features, result, label);
      RemoveDeal(i);
     }
  }

//+------------------------------------------------------------------+
//| Результат закрытой позиции по истории сделок (TP/SL и т.п.)      |
//+------------------------------------------------------------------+
bool GetClosedDealProfit(ulong position_ticket, double &profit)
  {
   if(!HistorySelectByPosition(position_ticket))
      return(false);

   int total = HistoryDealsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP);
      return(true);
     }
   return(false);
  }

//+------------------------------------------------------------------+
//| Формирование имени файла по интервалу теста                      |
//+------------------------------------------------------------------+
string GenerateFileName(string signal_type, datetime start, datetime end)
  {
   MqlDateTime s, e;
   TimeToStruct(start, s);
   TimeToStruct(end, e);
   string start_str = StringFormat("%04d%02d%02d", s.year, s.mon, s.day);
   string end_str   = StringFormat("%04d%02d%02d", e.year, e.mon, e.day);
   return StringFormat("data_%s_%s_%s.csv", signal_type, start_str, end_str);
  }

//+------------------------------------------------------------------+
//| Открытие CSV-файлов (Common\Files) на первой свече               |
//| Имя временное (start_start); финальное (start_end) формируется   |
//| в OnDeinit через FileMove.                                       |
//+------------------------------------------------------------------+
void EnsureFilesOpen()
  {
   if(test_start == 0)
      test_start = iTime(_Symbol, PERIOD_M15, 1);

   if(Inp_ExportBuy)
     {
      fnameBuy = GenerateFileName("BUY", test_start, test_start);
      if(!csvFileBuy.Open(fnameBuy, FILE_CSV|FILE_WRITE|FILE_ANSI|FILE_COMMON))
        {
         Print("Ошибка открытия файла: ", fnameBuy);
         return;
        }
      WriteHeader(csvFileBuy);
     }

   if(Inp_ExportSell)
     {
      fnameSell = GenerateFileName("SELL", test_start, test_start);
      if(!csvFileSell.Open(fnameSell, FILE_CSV|FILE_WRITE|FILE_ANSI|FILE_COMMON))
        {
         Print("Ошибка открытия файла: ", fnameSell);
         return;
        }
      WriteHeader(csvFileSell);
     }

   files_ok = true;
  }

//+------------------------------------------------------------------+
//| Закрытие позиций: обработка TP/SL и горизонта Inp_HoldBars       |
//+------------------------------------------------------------------+
void ProcessOpenDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         // позиция закрыта сервером (TP/SL) - результат по истории сделок
         double p = 0.0;
         int label = GetClosedDealProfit(deals[i].ticket, p) ? ((p > 0.0) ? 1 : 0) : 0;
         WriteDealRow(deals[i].type, deals[i].signal_time, deals[i].features, p, label);
         RemoveDeal(i);
         continue;
        }

      deals[i].held_bars++;
      if(deals[i].held_bars >= Inp_HoldBars)
        {
         double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
         int label = 0;
         if(trade.PositionClose(deals[i].ticket))
            label = (profit > 0.0) ? 1 : 0;
         // если закрыть не удалось - сделка не закрылась вовремя, флаг 0
         WriteDealRow(deals[i].type, deals[i].signal_time, deals[i].features, profit, label);
         RemoveDeal(i);
        }
     }
  }

//+------------------------------------------------------------------+
//| Детекция сигналов на последнем закрытом баре и открытие ордеров  |
//+------------------------------------------------------------------+
void DetectAndTrade()
  {
  int cnt = EXPORT_SEQ_LEN + 2;   // окно + сигнальные бары
  int min_need = EXPORT_SEQ_LEN + 2;
  double open[], close[], high[], low[];
   long   volume[];
   double ema8[], ema21[], rsi[], stochK[], stochD[], macd_main[], macd_signal[], atr[];

   // Копируем по максимуму, но ДОПУСКАЕМ меньше истории (накопление):
   // avail = сколько баров реально доступно (минимум по всем буферам).
  int avail = CopyOpen(_Symbol, PERIOD_M15, 0, cnt, open);
  int n = CopyClose(_Symbol, PERIOD_M15, 0, cnt, close);
  if(n < avail) avail = n;
   if(avail < min_need) return;
   n = CopyHigh(_Symbol, PERIOD_M15, 0, cnt, high);         if(n < avail) avail = n;
   n = CopyLow(_Symbol, PERIOD_M15, 0, cnt, low);           if(n < avail) avail = n;
   n = CopyTickVolume(_Symbol, PERIOD_M15, 0, cnt, volume); if(n < avail) avail = n;

   n = CopyBuffer(handles[0], 0, 0, cnt, ema8);        if(n < avail) avail = n;
   n = CopyBuffer(handles[1], 0, 0, cnt, ema21);       if(n < avail) avail = n;
   n = CopyBuffer(handles[2], 0, 0, cnt, rsi);         if(n < avail) avail = n;
   n = CopyBuffer(handles[3], 0, 0, cnt, stochK);      if(n < avail) avail = n;
   n = CopyBuffer(handles[3], 1, 0, cnt, stochD);      if(n < avail) avail = n;
   n = CopyBuffer(handles[4], 0, 0, cnt, macd_main);   if(n < avail) avail = n;
   n = CopyBuffer(handles[4], 1, 0, cnt, macd_signal); if(n < avail) avail = n;
   n = CopyBuffer(handles[5], 0, 0, cnt, atr);         if(n < avail) avail = n;

   if(avail < min_need) return;
   int avail_bars = avail;   // баров реально в буферах (индексы 0..avail-1)

   const int shift = 1;   // последний закрытый бар

   // --- Детекция сырых сигналов ---
   bool buy_signal = false, sell_signal = false;

   // EMA cross
   if(ema8[shift] > ema21[shift] && ema8[shift+1] <= ema21[shift+1])
      buy_signal = true;
   if(ema8[shift] < ema21[shift] && ema8[shift+1] >= ema21[shift+1])
      sell_signal = true;

   // RSI
   if(rsi[shift] > 38 && rsi[shift+1] <= 38)
      buy_signal = true;
   if(rsi[shift] < 62 && rsi[shift+1] >= 62)
      sell_signal = true;

   // Stochastic
   if(stochK[shift] > stochD[shift] && stochK[shift+1] <= stochD[shift+1] && stochK[shift] <= 30)
      buy_signal = true;
   if(stochK[shift] < stochD[shift] && stochK[shift+1] >= stochD[shift+1] && stochK[shift] >= 70)
      sell_signal = true;

   // MACD
   if(macd_main[shift] > macd_signal[shift] && macd_main[shift+1] <= macd_signal[shift+1])
      buy_signal = true;
   if(macd_main[shift] < macd_signal[shift] && macd_main[shift+1] >= macd_signal[shift+1])
      sell_signal = true;

   datetime signal_time = iTime(_Symbol, PERIOD_M15, shift);
   if(IsLastBarOfDay(signal_time))
      return;   // сигналы на последнем баре дня не открываем

   if(buy_signal && Inp_ExportBuy)  buy_signals++;
   if(sell_signal && Inp_ExportSell) sell_signals++;

   if(buy_signal && Inp_ExportBuy)
     {
      double features[];
      ArrayResize(features, total_features);
      BuildFeatureVector(shift, open, close, high, low, volume,
                         ema8, ema21, rsi, stochK, stochD,
                         macd_main, macd_signal, atr,
                         EXPORT_SEQ_LEN, avail_bars, features);

      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double sl  = (Inp_SL_Points > 0.0) ? ask - sl_price : 0.0;
      double tp  = (Inp_TP_Points > 0.0) ? ask + tp_price : 0.0;

      if(trade.Buy(Inp_LotSize, _Symbol, ask, sl, tp, "RNN_Data"))
        {
         ulong ticket = trade.ResultOrder();
         if(ticket > 0)
           {
            AddDeal(ticket, 0, signal_time, features);
            buy_trades++;
           }
        }
     }

   if(sell_signal && Inp_ExportSell)
     {
      double features[];
      ArrayResize(features, total_features);
      BuildFeatureVector(shift, open, close, high, low, volume,
                         ema8, ema21, rsi, stochK, stochD,
                         macd_main, macd_signal, atr,
                         EXPORT_SEQ_LEN, avail_bars, features);

      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl  = (Inp_SL_Points > 0.0) ? bid + sl_price : 0.0;
      double tp  = (Inp_TP_Points > 0.0) ? bid - tp_price : 0.0;

      if(trade.Sell(Inp_LotSize, _Symbol, bid, sl, tp, "RNN_Data"))
        {
         ulong ticket = trade.ResultOrder();
         if(ticket > 0)
           {
            AddDeal(ticket, 1, signal_time, features);
            sell_trades++;
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Добавление сделки в список отслеживания                          |
//+------------------------------------------------------------------+
void AddDeal(ulong ticket, int type, datetime signal_time, double &features[])
  {
   int idx = deals_count++;
   ArrayResize(deals, deals_count);
   deals[idx].ticket      = ticket;
   deals[idx].type        = type;
   deals[idx].signal_time = signal_time;
   deals[idx].held_bars   = 0;
   ArrayResize(deals[idx].features, total_features);
   ArrayCopy(deals[idx].features, features);
  }

//+------------------------------------------------------------------+
//| Удаление сделки из списка отслеживания                           |
//+------------------------------------------------------------------+
void RemoveDeal(int idx)
  {
   deals_count--;
   if(idx < deals_count)
     {
      deals[idx].ticket      = deals[deals_count].ticket;
      deals[idx].type        = deals[deals_count].type;
      deals[idx].signal_time = deals[deals_count].signal_time;
      deals[idx].held_bars   = deals[deals_count].held_bars;
      ArrayCopy(deals[idx].features, deals[deals_count].features);
     }
   ArrayResize(deals[deals_count].features, 0);
   ArrayResize(deals, deals_count);
  }

//+------------------------------------------------------------------+
//| Запись строки в соответствующий файл                              |
//+------------------------------------------------------------------+
void WriteDealRow(int type, datetime time, double &features[], double result, int label)
  {
   if(type == 0)
     {
      WriteRowToCSV(csvFileBuy, time, "BUY", features, result, label);
      if(label == 0) buy_label0++;
      else           buy_label1++;
     }
   else
     {
      WriteRowToCSV(csvFileSell, time, "SELL", features, result, label);
      if(label == 0) sell_label0++;
      else           sell_label1++;
     }
  }

//+------------------------------------------------------------------+
//| Запись заголовка в CSV: datetime,signal,lag0_*,...,lag19_*,      |
//|   lag0 - сигнальный бар (самый свежий), порядок колонок          |
//|   совпадает с порядком признаков в векторе.                      |
//+------------------------------------------------------------------+
void WriteHeader(CFileTxt &file)
  {
   file.WriteString("datetime,signal");
  string names[EXPORT_FEATURES] = {"open","high","low","close","volume",
                        "ema8","ema21","rsi","stoch_k","stoch_d",
                        "macd_main","macd_signal","atr","sin_hour",
                        "cos_hour","day_of_week","direction","body_range"};
  for(int lag = 0; lag < EXPORT_SEQ_LEN; lag++)
     {
    for(int i = 0; i < EXPORT_FEATURES; i++)
         file.WriteString("," + "lag" + IntegerToString(lag) + "_" + names[i]);
     }
   file.WriteString(",result,label\n");
  }

//+------------------------------------------------------------------+
//| Заполнение OHLC, индикаторов и дополнительных признаков бара.   |
//+------------------------------------------------------------------+
void FillBarFeatures(int bar,
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
                     double &out[])
  {
  out[0]  = open[bar];
  out[1]  = high[bar];
  out[2]  = low[bar];
  out[3]  = close[bar];
  out[4]  = (double)volume[bar];
  out[5]  = ema8[bar];
  out[6]  = ema21[bar];
  out[7]  = rsi[bar];
  out[8]  = stochK[bar];
  out[9]  = stochD[bar];
  out[10] = macd_main[bar];
  out[11] = macd_signal[bar];
  out[12] = atr[bar];

  datetime bar_time = iTime(_Symbol, PERIOD_M15, bar);
  MqlDateTime dt;
  TimeToStruct(bar_time, dt);
  double hour = dt.hour + dt.min / 60.0;
  out[13] = MathSin(2.0 * M_PI * hour / 24.0);
  out[14] = MathCos(2.0 * M_PI * hour / 24.0);
  out[15] = (double)dt.day_of_week / 6.0;
  out[16] = (close[bar] > open[bar]) ? 1.0 : -1.0;
  double range = high[bar] - low[bar];
  double body  = MathAbs(close[bar] - open[bar]);
  out[17] = (range > 0.0) ? body / range : 0.0;
  }

//+------------------------------------------------------------------+
//| Формирование вектора признаков (seq_len x 18).                   |
//| Внимание: f0..f17 - сигнальный бар (самый свежий в окне),        |
//| далее - более старые бары (порядок: новое -> старое).            |
//| Для формирования строки используются только реальные бары.       |
//+------------------------------------------------------------------+
void BuildFeatureVector(int signal_shift,
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
                        int seq_len,
                        int avail_bars,
                        double &features[])
  {
  double bar_feat[EXPORT_FEATURES];
   // сколько реальных баров попадает в окно (от сигнала вглубь)
   int n_real = avail_bars - signal_shift;
   if(n_real > seq_len)
      n_real = seq_len;

  if(n_real < seq_len)
    return;

   int idx = 0;
   for(int k = 0; k < seq_len; k++)
     {
    FillBarFeatures(signal_shift + k, open, close, high, low, volume,
               ema8, ema21, rsi, stochK, stochD,
               macd_main, macd_signal, atr, bar_feat);
    for(int f = 0; f < EXPORT_FEATURES; f++)
      features[idx++] = bar_feat[f];
     }
  }

//+------------------------------------------------------------------+
//| Запись строки в CSV                                              |
//+------------------------------------------------------------------+
void WriteRowToCSV(CFileTxt &file, datetime time, string signal, double &features[], double result, int label)
  {
   string row = StringFormat("%s,%s", TimeToString(time), signal);
   int size = ArraySize(features);
   for(int i=0; i<size; i++)
      row += StringFormat(",%.6f", features[i]);
   row += StringFormat(",%.2f,%d\n", result, label);
   file.WriteString(row);
   rows_written++;
  }
//+------------------------------------------------------------------+
