//+------------------------------------------------------------------+
//|                                                   RNNFilter.mq5 |
//|                          Copyright 2026, Bondarev A.            |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Bondarev A."
#property link      "https://www.mql5.com"
#property version   "1.04"
//+------------------------------------------------------------------+
//|                    RNN Filter Expert EA                          |
//|   Торговля через фильтрацию сигналов GRU-нейросетью (ONNX)       |
//|   BUY-сигнал -> сеть BUY, SELL-сигнал -> сеть SELL.              |
//|   Решение принимается по уверенности своей сети (P(прибыль))     |
//|   Система серий с шагами, кулдауном, трейлингом и безубытком     |
//|                                                                  |
//|   Модули:                                                        |
//|   - RNN_ONNX.mqh     NN-фильтр (ONNX), активен при               |
//|                       Inp_UseNNFilter == true                    |
//|   - RNNFilter_News.mqh фильтр новостей (календарь), активен при  |
//|                       Inp_NewsFilterEnable == true               |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include "RNN_ONNX.mqh"
#include "RNNFilter_News.mqh"
//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
// --- Включение ---
// Направления: true = сигналы учитываются, false = игнорируются
input bool   Inp_Enabled          = true;      // Включить советник
input bool   Inp_BuySignal        = true;      // Отключить BUY сигналы 
input bool   Inp_SellSignal       = true;      // Отключить SELL сигналы 
input bool   Inp_UseNNFilter      = true;      // Использовать фильтр НС (ONNX)

// --- Пороги уверенности НС (каждая сеть решает ТОЛЬКО своё направление) ---
input double Inp_BuyConfThreshold  = 0.7;       // НС BUY: порог уверенности (P(прибыль) >= порог)
input double Inp_SellConfThreshold = 0.7;       // НС SELL: порог уверенности (P(прибыль) >= порог)
input int    Inp_NNProfitClass     = 1;        // Класс "прибыльной" сделки (0/1, см. learn.py)

// --- Пути к ONNX-моделям (относительно MQL5\Files) ---
input string Inp_BuyModelFile     = "RNN\\buy_model.onnx";   // Файл модели BUY
input string Inp_SellModelFile    = "RNN\\sell_model.onnx";  // Файл модели SELL

// --- Торговые настройки ---
input double Inp_LotSize          = 0.01;      // Базовый объём лота
input double Inp_LotMultiplier    = 1.5;       // Множитель шага серии
input int    Inp_Magic            = 777111;    // Магик-номер

input double Inp_SL_Points        = 150.0;     // Стоп-лосс (пункты), 0 = без SL
input double Inp_RR               = 3.0;       // Риск/прибыль: TP = RR × SL (1:3), 0 = без TP

// --- Трейлинг ---
input bool   Inp_TrailingEnable   = false;     // Включить трейлинг
input double Inp_TrailingPoints   = 100.0;     // Трейлинг-дистанция (пункты)
input double Inp_TrailingStartPoints = 50.0;   // Стартовый профит для трейлинга (пункты)

// --- Безубыток ---
input bool   Inp_BreakEvenEnable  = false;     // Включить безубыток
input double Inp_BreakEvenPoints  = 50.0;      // Безубыток уровень (пункты)
input double Inp_BEOffsetPoints   = 20.0;      // Смещение от цены (пункты)

// --- Система серий ---
input int    Inp_HoldBars         = 20;        // Макс. горизонт удержания (бары)
// Inp_SequenceLen удалён: окно нормализации = NN_NORM_WINDOW, окно входа = NN_SEQ_LEN
// (оба из RNN_Scaler.mqh, генерируется export_onnx.py из learn.py)
input int    Inp_SignalCooldown   = 3;         // Кулдаун после сигнала (бары)
input int    Inp_MaxSeriesSteps   = 4;         // Макс. шагов в серии
input int    Inp_SeriesBarLimit   = 30;        // Макс. баров в серии

// --- Фильтр спреда ---
input bool   Inp_SpreadEnable     = false;     // Включить фильтр спреда
input int    Inp_MaxSpreadPoints  = 30;        // Макс. спред (пункты)

// --- Фильтр новостей ---
input bool   Inp_NewsFilterEnable = false;     // Включить фильтр новостей
input int    Inp_NewsEventTime    = 3;         // Важность события (1-3)
input int    Inp_NewsWindowBefore = 60;        // Окно до новости (минуты)
input int    Inp_NewsWindowAfter  = 60;        // Окно после новости (минуты)

// --- Пользовательский критерий оптимизации (winrate) ---
//     Подключается ПОСЛЕ входных параметров: использует Inp_Magic.
//     В тестере выберите критерий "Пользовательский максимум" (Custom max).
#include "TargetWinRateCriterion.mqh"
//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CTrade                  trade;

// Индикаторные хендлы: 0=EMA8, 1=EMA21, 2=RSI, 3=Stoch, 4=MACD, 5=ATR
int                     handles[6];

// Структура записи сделки
struct DealRecord
  {
   ulong         ticket;
   int           type;              // 0=BUY, 1=SELL
   datetime      signal_time;
   int           held_bars;
   int           series_step;
   double        entry_price;
   double        sl;
   double        tp;
  };

DealRecord            deals[];
int                   deals_count = 0;

// Структура серии
struct SeriesInfo
  {
   bool          active;
   int           step;
   datetime      first_bar;
   datetime      last_signal_bar;
   int           signals_total;
  };

SeriesInfo          buySeries;
SeriesInfo          sellSeries;

// Прочее
int                   nn_buy_seq_len = 0;
int                   g_series_bar_limit = 0;   // лимит серии (корректируется в OnInit)
int                   nn_sell_seq_len = 0;
datetime              last_bar_time = 0;


double                sl_price = 0.0;
double                trail_points = 0.0;
double                trail_start = 0.0;
double                be_points = 0.0;
double                be_offset = 0.0;
string                g_price_fmt = "%.2f";   // формат цены в логах (по _Digits символа)

// Буферы для индикаторов и рыночных данных
double                close_buf[], high_buf[], low_buf[];
long                  volume_buf[];
double                ema8_buf[], ema21_buf[], rsi_buf[];
double                stochK_buf[], stochD_buf[];
double                macd_main_buf[], macd_signal_buf[], atr_buf[];
//+------------------------------------------------------------------+
//| Инициализация                                                    |
//+------------------------------------------------------------------+
int OnInit()
  {
   // --- Валидация параметров ---
   if(Inp_HoldBars < 1 ||
      Inp_LotSize <= 0 || Inp_MaxSeriesSteps < 1 ||
      Inp_SignalCooldown < 0)

     {
      Print("Ошибка параметров: HoldBars>=1, LotSize>0, MaxSteps>=1, Cooldown>=0");
      return(INIT_FAILED);
     }

   // Серия не должна завершаться раньше, чем закроется первая сделка серии:
   // если SeriesBarLimit < HoldBars — используем HoldBars (вместо отказа).
   g_series_bar_limit = Inp_SeriesBarLimit;
   if(g_series_bar_limit < Inp_HoldBars)
     {
      PrintFormat("Внимание: Inp_SeriesBarLimit=%d < Inp_HoldBars=%d, лимит серии поднят до %d",
                  Inp_SeriesBarLimit, Inp_HoldBars, Inp_HoldBars);
      g_series_bar_limit = Inp_HoldBars;
     }

   // --- Пункты: минимум изменения цены (point) ---
   
   g_price_fmt = StringFormat("%%.%df", _Digits);   // формат цен в логах (по _Digits)

   // --- Валидация порогов НС ---
   if(Inp_BuyConfThreshold <= 0.0 ||
      Inp_SellConfThreshold <= 0.0 ||
      false)   // Inp_NNProfitClass: 0 → P0, иначе → P1 (валидация отключена для оптимизации)
     {
      Print("Ошибка порогов НС: пороги уверенности должны быть > 0");
      return(INIT_FAILED);
     }
   // --- Окно входа НС из конфига (RNN_Scaler.mqh), не задаётся вручную ---
   nn_buy_seq_len = NN_ONNX_GetSeqLen(0);
   nn_sell_seq_len = NN_ONNX_GetSeqLen(1);
   if(nn_buy_seq_len < 1 || nn_buy_seq_len > 60 ||
      nn_sell_seq_len < 1 || nn_sell_seq_len > 60)
     {
      PrintFormat("Ошибка: некорректные окна входа НС BUY=%d SELL=%d (ожидалось 1..60)",
                  nn_buy_seq_len, nn_sell_seq_len);
      return(INIT_FAILED);
     }


   sl_price = Inp_SL_Points * _Point;
   trail_points = Inp_TrailingPoints * _Point;
   trail_start = Inp_TrailingStartPoints * _Point;
   be_points = Inp_BreakEvenPoints * _Point;
   be_offset = Inp_BEOffsetPoints * _Point;

   // --- Создаём индикаторные хендлы ---
   handles[0] = iMA(_Symbol, PERIOD_M15, 8,  0, MODE_EMA, PRICE_CLOSE);
   handles[1] = iMA(_Symbol, PERIOD_M15, 21, 0, MODE_EMA, PRICE_CLOSE);
   handles[2] = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   handles[3] = iStochastic(_Symbol, PERIOD_M15, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   handles[4] = iMACD(_Symbol, PERIOD_M15, 8, 17, 6, PRICE_CLOSE);
   handles[5] = iATR(_Symbol, PERIOD_M15, 14);

   for(int i = 0; i < 6; i++)
     {
      if(handles[i] == INVALID_HANDLE)
        {
         PrintFormat("Ошибка создания индикатора %d (код: %d)", i, GetLastError());
         return(INIT_FAILED);
        }
     }

   // --- Настраиваем торговый объект ---
   trade.SetExpertMagicNumber(Inp_Magic);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFillingBySymbol(_Symbol);

   // --- Инициализируем ONNX-модели ---
   if(Inp_UseNNFilter)
     {
      if(!NN_ONNX_Init(Inp_BuyModelFile, Inp_SellModelFile))
        {
         Print("RNNFilter: не удалось загрузить ONNX-модели. Файлы должны лежать в Common\\Files\\RNN\\ (для тестера) или MQL5\\Files\\RNN\\ (для графика)");
         return(INIT_FAILED);
        }
     }

   // --- Инициализируем серии ---
   ResetSeries(buySeries);
   ResetSeries(sellSeries);

   PrintFormat("RNNFilter инициализирован. NN: ONNX (%s | %s), пороги BUY=%.2f SELL=%.2f (класс прибыли %d), окна НС BUY=%d SELL=%d баров, RR: 1:%.1f, SL: %.1f п., Hold: %d, MaxSteps: %d, Cooldown: %d",
               Inp_BuyModelFile, Inp_SellModelFile,
               Inp_BuyConfThreshold, Inp_SellConfThreshold, Inp_NNProfitClass,
               nn_buy_seq_len, nn_sell_seq_len,
               Inp_RR, Inp_SL_Points, Inp_HoldBars, Inp_MaxSeriesSteps, Inp_SignalCooldown);
   PrintFormat("RNNFilter: сигналы BUY=%s, SELL=%s", Inp_BuySignal ? "ВКЛ" : "ВЫКЛ", Inp_SellSignal ? "ВКЛ" : "ВЫКЛ");
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Деинициализация                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Закрываем все оставшиеся позиции
   CloseAllDeals();

   // Освобождаем ONNX-сессии
   if(Inp_UseNNFilter)
      NN_ONNX_Deinit();

   // Освобождаем индикаторные хендлы
   for(int i = 0; i < 6; i++)
     {
      if(handles[i] != INVALID_HANDLE)
         IndicatorRelease(handles[i]);
     }

   PrintFormat("RNNFilter деинициализирован. Сделок обработано: %d", deals_count);
  }
//+------------------------------------------------------------------+
//| Обработка каждого тика                                           |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Глобальный выключатель
   if(!Inp_Enabled)
      return;

   datetime cur_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(cur_bar == last_bar_time)
      return;  // Та же самая свеча — ничего не делаем

   last_bar_time = cur_bar;

   // На последнем баре дня закрываем все позиции
   if(IsLastBarOfDay(cur_bar))
     {
      CloseAllDeals();
      return;  // новые сигналы в конце дня не открываем
     }

   // --- Обрабатываем открытые сделки ---
   ProcessOpenDeals();

   // --- Трейлинг и безубыток ---
   if(Inp_TrailingEnable)
      TrailingPositions();
   if(Inp_BreakEvenEnable)
      MoveToBreakEven();

   // --- Фильтры (спред, новости) ---
   if(Inp_SpreadEnable && !IsSpreadOK())
      return;
   if(Inp_NewsFilterEnable &&
      IsNewsBlocked(Inp_NewsEventTime, Inp_NewsWindowBefore, Inp_NewsWindowAfter))
      return;

   // --- Проверяем сигналы ---
   CheckSignals();
  }
//+------------------------------------------------------------------+
//| Сброс серии                                                      |
//+------------------------------------------------------------------+
void ResetSeries(SeriesInfo &series)
  {
   series.active          = false;
   series.step            = 0;
   series.first_bar       = 0;
   series.last_signal_bar = 0;
   series.signals_total   = 0;
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
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         RemoveDeal(i);
         continue;
        }

      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      if(trade.PositionClose(deals[i].ticket))
        {
         PrintFormat("Закрыта сделка #%I64u, профит: %.2f", deals[i].ticket, profit);
        }
      else
        {
         PrintFormat("Ошибка закрытия сделки #%I64u", deals[i].ticket);
        }
      RemoveDeal(i);
     }
  }
//+------------------------------------------------------------------+
//| Обработка открытых сделок                                        |
//+------------------------------------------------------------------+
void ProcessOpenDeals()
  {
   for(int i = deals_count - 1; i >= 0; i--)
     {
      // Проверяем, закрыта ли позиция сервером
      if(!PositionSelectByTicket(deals[i].ticket))
        {
         RemoveDeal(i);
         continue;
        }

      deals[i].held_bars++;

      // Проверяем горизонт удержания
      if(deals[i].held_bars >= Inp_HoldBars)
        {
         if(trade.PositionClose(deals[i].ticket))
           {
            PrintFormat("Досрочное закрытие #%I64u по горизонту (%d баров)",
                        deals[i].ticket, deals[i].held_bars);
           }
         RemoveDeal(i);
         continue;
        }
     }
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
      deals[idx].series_step = deals[deals_count].series_step;
      deals[idx].entry_price = deals[deals_count].entry_price;
      deals[idx].sl          = deals[deals_count].sl;
      deals[idx].tp          = deals[deals_count].tp;
     }
   ArrayResize(deals, deals_count);
  }
//+------------------------------------------------------------------+
//| Трейлинг позиций                                                 |
//+------------------------------------------------------------------+
void TrailingPositions()
  {
   for(int i = 0; i < deals_count; i++)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
         continue;

      double pos_sl = PositionGetDouble(POSITION_SL);
      double entry  = PositionGetDouble(POSITION_PRICE_OPEN);

      if(deals[i].type == 0) // BUY
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         // Проверяем минимальный профит для старта трейлинга (в цене)
         if((bid - entry) < trail_start)
            continue;

         double new_sl = bid - trail_points;
         // Новый SL выше текущего и выше цены входа
         if(new_sl > pos_sl && new_sl > entry)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
           }
        }
      else // SELL
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         // Проверяем минимальный профит для старта трейлинга (в цене)
         if((entry - ask) < trail_start)
            continue;

         double new_sl = ask + trail_points;
         // Новый SL ниже текущего и ниже цены входа
         if((new_sl < pos_sl || pos_sl == 0) && new_sl < entry)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Перевод в безубыток                                              |
//+------------------------------------------------------------------+
void MoveToBreakEven()
  {
   for(int i = 0; i < deals_count; i++)
     {
      if(!PositionSelectByTicket(deals[i].ticket))
         continue;

      double pos_sl = PositionGetDouble(POSITION_SL);
      double entry  = PositionGetDouble(POSITION_PRICE_OPEN);

      if(deals[i].type == 0) // BUY
        {
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         // Проверяем, достигли ли уровня безубытка
         if((bid - entry) < be_points)
            continue;

         double new_sl = entry + be_offset;
         // Перемещаем, если SL ещё не установлен или ниже
         if(pos_sl < new_sl)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
           }
        }
      else // SELL
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         // Проверяем, достигли ли уровня безубытка
         if((entry - ask) < be_points)
            continue;

         double new_sl = entry - be_offset;
         // Перемещаем, если SL ещё не установлен или выше
         if(pos_sl == 0 || pos_sl > new_sl)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
           }
        }
     }
  }
//+------------------------------------------------------------------+
//| Проверка спреда                                                  |
//+------------------------------------------------------------------+
bool IsSpreadOK()
  {
   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   return(spread <= Inp_MaxSpreadPoints);
  }
//+------------------------------------------------------------------+
//| Нормализация объёма лота                                         |
//+------------------------------------------------------------------+
double NormalizeLot(double lot)
  {
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(lot_step <= 0.0)
      lot_step = 0.01;

   lot = MathMax(MathMin(lot, max_lot), min_lot);
   lot = MathRound(lot / lot_step) * lot_step;
   return(NormalizeDouble(lot, 8));
  }
//+------------------------------------------------------------------+
//| Проверка сигналов и открытие сделок                              |
//+------------------------------------------------------------------+
void CheckSignals()
  {
   int cnt = NN_MAX_NORM_WINDOW + 3;
   int min_need = MathMax(NN_BUY_SEQ_LEN, NN_SELL_SEQ_LEN) + 2;

   // Копируем по максимуму, но ДОПУСКАЕМ меньше истории (накопление):
   // avail = сколько баров реально доступно (минимум по всем буферам).
   int avail = CopyClose(_Symbol, PERIOD_M15, 0, cnt, close_buf);
   if(avail < min_need) return;
   int n;
   n = CopyHigh(_Symbol, PERIOD_M15, 0, cnt, high_buf);         if(n < avail) avail = n;
   n = CopyLow(_Symbol, PERIOD_M15, 0, cnt, low_buf);           if(n < avail) avail = n;
   n = CopyTickVolume(_Symbol, PERIOD_M15, 0, cnt, volume_buf); if(n < avail) avail = n;

   n = CopyBuffer(handles[0], 0, 0, cnt, ema8_buf);        if(n < avail) avail = n;
   n = CopyBuffer(handles[1], 0, 0, cnt, ema21_buf);       if(n < avail) avail = n;
   n = CopyBuffer(handles[2], 0, 0, cnt, rsi_buf);         if(n < avail) avail = n;
   n = CopyBuffer(handles[3], 0, 0, cnt, stochK_buf);      if(n < avail) avail = n;
   n = CopyBuffer(handles[3], 1, 0, cnt, stochD_buf);      if(n < avail) avail = n;
   n = CopyBuffer(handles[4], 0, 0, cnt, macd_main_buf);   if(n < avail) avail = n;
   n = CopyBuffer(handles[4], 1, 0, cnt, macd_signal_buf); if(n < avail) avail = n;
   n = CopyBuffer(handles[5], 0, 0, cnt, atr_buf);         if(n < avail) avail = n;

   if(avail < min_need) return;
   int avail_bars = avail;   // баров реально в буферах (индексы 0..avail-1)

   const int shift = 1;  // последний закрытый бар
   datetime signal_time = iTime(_Symbol, PERIOD_M15, shift);
   if(IsLastBarOfDay(signal_time))
      return;  // сигналы на последнем баре дня не открываем

   // --- Детекция сырых сигналов ---
   bool buy_signal = false, sell_signal = false;

   // EMA cross
   if(ema8_buf[shift] > ema21_buf[shift] && ema8_buf[shift + 1] <= ema21_buf[shift + 1])
      buy_signal = true;
   if(ema8_buf[shift] < ema21_buf[shift] && ema8_buf[shift + 1] >= ema21_buf[shift + 1])
      sell_signal = true;

   // RSI
   if(rsi_buf[shift] > 38 && rsi_buf[shift + 1] <= 38)
      buy_signal = true;
   if(rsi_buf[shift] < 62 && rsi_buf[shift + 1] >= 62)
      sell_signal = true;

   // Stochastic
   if(stochK_buf[shift] > stochD_buf[shift] &&
      stochK_buf[shift + 1] <= stochD_buf[shift + 1] && stochK_buf[shift] <= 30)
      buy_signal = true;
   if(stochK_buf[shift] < stochD_buf[shift] &&
      stochK_buf[shift + 1] >= stochD_buf[shift + 1] && stochK_buf[shift] >= 70)
      sell_signal = true;

   // MACD
   if(macd_main_buf[shift] > macd_signal_buf[shift] &&
      macd_main_buf[shift + 1] <= macd_signal_buf[shift + 1])
      buy_signal = true;
   if(macd_main_buf[shift] < macd_signal_buf[shift] &&
      macd_main_buf[shift + 1] >= macd_signal_buf[shift + 1])
      sell_signal = true;

   // --- Прогон сетей ONNX ТОЛЬКО по направлениям сигналов ---
   //    BUY-сигнал -> сеть BUY, SELL-сигнал -> сеть SELL.
   //    Решение по каждому направлению принимает ТОЛЬКО своя сеть.
   double conf_buy = 0.0, conf_sell = 0.0;
   bool   nn_active = (Inp_UseNNFilter && NN_ONNX_IsReady());
   double buy_features[];
   double sell_features[];

   if(nn_active && buy_signal && Inp_BuySignal)
     {
      ArrayResize(buy_features, NN_BUY_NORM_WINDOW * NN_FEATURES);
      NN_BuildFeatures(shift, close_buf, high_buf, low_buf, volume_buf,
                       ema8_buf, ema21_buf, rsi_buf,
                       stochK_buf, stochD_buf,
                       macd_main_buf, macd_signal_buf, atr_buf,
                       NN_BUY_NORM_WINDOW, avail_bars, buy_features);

      conf_buy = NN_ONNX_Predict(0, buy_features, Inp_NNProfitClass);
      if(conf_buy < 0.0)
        {
         PrintFormat("RNNFilter: ошибка ONNX-инференса BUY (код %d)", GetLastError());
         return;
        }
     }

   if(nn_active && sell_signal && Inp_SellSignal)
     {
      ArrayResize(sell_features, NN_SELL_NORM_WINDOW * NN_FEATURES);
      NN_BuildFeatures(shift, close_buf, high_buf, low_buf, volume_buf,
                          ema8_buf, ema21_buf, rsi_buf,
                          stochK_buf, stochD_buf,
                          macd_main_buf, macd_signal_buf, atr_buf,
                          NN_SELL_NORM_WINDOW, avail_bars, sell_features);
      conf_sell = NN_ONNX_Predict(1, sell_features, Inp_NNProfitClass);
      if(conf_sell < 0.0)
        {
         PrintFormat("RNNFilter: ошибка ONNX-инференса SELL (код %d)", GetLastError());
         return;
        }
     }

   // --- Обработка BUY сигнала ---
   if(buy_signal && Inp_BuySignal)
      ProcessBuySignal(signal_time, conf_buy);

   // --- Обработка SELL сигнала ---
   if(sell_signal && Inp_SellSignal)
      ProcessSellSignal(signal_time, conf_sell);
  }
//+------------------------------------------------------------------+
//| Обработка BUY сигнала (проверка НС -> исполнение)                |
//+------------------------------------------------------------------+
void ProcessBuySignal(datetime signal_time, double conf_buy)
  {
   if(Inp_UseNNFilter)
     {
      // Решение принимает ТОЛЬКО сеть BUY по своей уверенности P(прибыль)
      if(conf_buy < Inp_BuyConfThreshold)
        {
         PrintFormat("BUY отклонён НС: confidence=%.4f (нужно >= %.2f)",
                     conf_buy, Inp_BuyConfThreshold);
         return;
        }
     }

   ExecuteBuySignal(signal_time, conf_buy);
  }
//+------------------------------------------------------------------+
//| Исполнение BUY по подтверждённому сигналу                        |
//+------------------------------------------------------------------+
void ExecuteBuySignal(datetime signal_time, double conf_buy)
  {
   // --- Проверяем серию ---
   if(!buySeries.active)
     {
      // Новая серия
      if(IsInCooldown(buySeries, signal_time))
         return;  // кулдаун

      StartNewSeries(buySeries, signal_time, 1);
     }
   else
     {
      // Активная серия
      if(IsInCooldown(buySeries, signal_time))
         return;

      // Проверяем лимит баров
      if((signal_time - buySeries.first_bar) / PeriodSeconds(PERIOD_M15) >= g_series_bar_limit)
        {
         ResetSeries(buySeries);
         return;
        }

      // Инкрементируем шаг
      int new_step = buySeries.step + 1;
      if(new_step > Inp_MaxSeriesSteps)
        {
         ResetSeries(buySeries);  // серия исчерпала себя
         return;
        }

      StartNewSeries(buySeries, signal_time, new_step);
     }

   // --- Открытие сделки BUY ---
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl  = (Inp_SL_Points > 0.0) ? ask - sl_price : 0.0;
   double tp  = (sl > 0.0 && Inp_RR > 0.0) ? ask + Inp_RR * (ask - sl) : 0.0;
   double lot = NormalizeLot(Inp_LotSize * MathPow(Inp_LotMultiplier, buySeries.step - 1));

   string cmt = "RNN_Buy";
   if(Inp_UseNNFilter)
      cmt = StringFormat("B%.2f", conf_buy);

   if(trade.Buy(lot, _Symbol, ask, sl, tp, cmt))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, 0, signal_time, buySeries.step, ask, sl, tp);
         PrintFormat("BUY #%I64u | Step %d/%d | Lot %.2f | Ask " + g_price_fmt + " | SL " + g_price_fmt + " | TP " + g_price_fmt + " | NN conf=%.2f",
                     ticket, buySeries.step, Inp_MaxSeriesSteps, lot, ask, sl, tp, conf_buy);
        }
     }
  }
//+------------------------------------------------------------------+
//| Обработка SELL сигнала (проверка НС -> исполнение)               |
//+------------------------------------------------------------------+
void ProcessSellSignal(datetime signal_time, double conf_sell)
  {
   if(Inp_UseNNFilter)
     {
      // Решение принимает ТОЛЬКО сеть SELL по своей уверенности P(прибыль)
      if(conf_sell < Inp_SellConfThreshold)
        {
         PrintFormat("SELL отклонён НС: confidence=%.4f (нужно >= %.2f)",
                     conf_sell, Inp_SellConfThreshold);
         return;
        }
     }

   ExecuteSellSignal(signal_time, conf_sell);
  }
//+------------------------------------------------------------------+
//| Исполнение SELL по подтверждённому сигналу                       |
//+------------------------------------------------------------------+
void ExecuteSellSignal(datetime signal_time, double conf_sell)
  {
   // --- Проверяем серию ---
   if(!sellSeries.active)
     {
      if(IsInCooldown(sellSeries, signal_time))
         return;

      StartNewSeries(sellSeries, signal_time, 1);
     }
   else
     {
      if(IsInCooldown(sellSeries, signal_time))
         return;

      if((signal_time - sellSeries.first_bar) / PeriodSeconds(PERIOD_M15) >= g_series_bar_limit)
        {
         ResetSeries(sellSeries);
         return;
        }

      int new_step = sellSeries.step + 1;
      if(new_step > Inp_MaxSeriesSteps)
        {
         ResetSeries(sellSeries);
         return;
        }

      StartNewSeries(sellSeries, signal_time, new_step);
     }

   // --- Открытие сделки SELL ---
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl  = (Inp_SL_Points > 0.0) ? bid + sl_price : 0.0;
   double tp  = (sl > 0.0 && Inp_RR > 0.0) ? bid - Inp_RR * (sl - bid) : 0.0;
   double lot = NormalizeLot(Inp_LotSize * MathPow(Inp_LotMultiplier, sellSeries.step - 1));

   string cmt = "RNN_Sell";
   if(Inp_UseNNFilter)
      cmt = StringFormat("S%.2f", conf_sell);

   if(trade.Sell(lot, _Symbol, bid, sl, tp, cmt))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, 1, signal_time, sellSeries.step, bid, sl, tp);
         PrintFormat("SELL #%I64u | Step %d/%d | Lot %.2f | Bid " + g_price_fmt + " | SL " + g_price_fmt + " | TP " + g_price_fmt + " | NN conf=%.2f",
                     ticket, sellSeries.step, Inp_MaxSeriesSteps, lot, bid, sl, tp, conf_sell);
        }
     }
  }
//+------------------------------------------------------------------+
//| Добавление сделки в список отслеживания                          |
//+------------------------------------------------------------------+
void AddDeal(ulong ticket, int type, datetime signal_time, int step,
             double entry, double sl, double tp)
  {
   int idx = deals_count++;
   ArrayResize(deals, deals_count);
   deals[idx].ticket      = ticket;
   deals[idx].type        = type;
   deals[idx].signal_time = signal_time;
   deals[idx].held_bars   = 0;
   deals[idx].series_step = step;
   deals[idx].entry_price = entry;
   deals[idx].sl          = sl;
   deals[idx].tp          = tp;
  }
//+------------------------------------------------------------------+
//| Проверка кулдауна                                                |
//+------------------------------------------------------------------+
bool IsInCooldown(SeriesInfo &series, datetime current_time)
  {
   if(!series.active)
      return(false);

   int bars_elapsed = (int)((current_time - series.last_signal_bar) / PeriodSeconds(PERIOD_M15));
   return(bars_elapsed < Inp_SignalCooldown);
  }
//+------------------------------------------------------------------+
//| Старт/обновление серии                                            |
//+------------------------------------------------------------------+
void StartNewSeries(SeriesInfo &series, datetime signal_time, int step)
  {
   series.active          = true;
   series.step            = step;
   series.first_bar       = series.first_bar == 0 ? signal_time : series.first_bar;
   series.last_signal_bar = signal_time;
   series.signals_total++;
  }
//+------------------------------------------------------------------+
