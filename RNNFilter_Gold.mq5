//+------------------------------------------------------------------+
//|                                                   RNNFilter.mq5 |
//|                          Copyright 2026, Bondarev A.            |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Bondarev A."
#property link      "https://www.mql5.com"
#property version   "1.05"
//+------------------------------------------------------------------+
//|                    RNN Filter Expert EA                          |
//|   Торговля через фильтрацию сигналов GRU-нейросетью (ONNX)       |
//|   ВСЕГДА только ОДНА активная сделка.                            |
//|   Спрашиваем ОБЕ сети:                                           |
//|     - BUY проходит, если BUY-сеть уверена в покупке И SELL-сеть  |
//|       НЕ уверена в продаже;                                      |
//|     - SELL проходит, если SELL-сеть уверена в продаже И BUY-сеть |
//|       НЕ уверена в покупке.                                      |
//|   В строку состояния (Comment) выводятся confidence обеих сетей. |
//|   Серии и множители лота удалены — фиксированный лот.            |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include "RNN_ONNX.mqh"
#include "RNNFilter_News.mqh"
//+------------------------------------------------------------------+
//| Входные параметры                                                |
//+------------------------------------------------------------------+
// --- Включение ---
input bool   Inp_Enabled          = true;      // Включить советник
input bool   Inp_BuySignal        = true;      // Учитывать BUY сигналы
input bool   Inp_SellSignal       = true;      // Учитывать SELL сигналы
input bool   Inp_UseNNFilter      = true;      // Использовать фильтр НС (ONNX)
input bool   Inp_TradeWOIndicators = false;    // Торговать БЕЗ индикаторных сигналов (только по сетям)

// --- Пороги уверенности обеих сетей ---
input double Inp_BuyConfThreshold  = 0.7;      // BUY-сеть: уверена в покупке, если conf >= порог
input double Inp_SellConfThreshold = 0.7;      // SELL-сеть: уверена в продаже, если conf >= порог
input int    Inp_NNProfitClass     = 1;        // Класс "прибыльной" сделки (0/1)

// --- Пути к ONNX-моделям (относительно Common\Files или MQL5\Files) ---
input string Inp_BuyModelFile     = "RNN_GOLD\\buy_model.onnx";   // Файл модели BUY
input string Inp_SellModelFile    = "RNN_GOLD\\sell_model.onnx";  // Файл модели SELL

// --- Торговые настройки (фиксированный лот, без серий/множителей) ---
input double Inp_LotSize          = 0.01;      // Объём лота (фиксированный)
input int    Inp_Magic            = 777111;    // Магик-номер

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
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

// --- Удержание ---
input int    Inp_HoldBars         = 20;        // Макс. горизонт удержания (бары)

// --- Фильтр спреда ---
input bool   Inp_SpreadEnable     = false;     // Включить фильтр спреда
input int    Inp_MaxSpreadPoints  = 30;        // Макс. спред (пункты)

// --- Фильтр новостей ---
input bool   Inp_NewsFilterEnable = false;     // Включить фильтр новостей
input int    Inp_NewsEventTime    = 3;         // Важность события (1-3)
input int    Inp_NewsWindowBefore = 60;        // Окно до новости (минуты)
input int    Inp_NewsWindowAfter  = 60;        // Окно после новости (минуты)

// --- Пользовательский критерий оптимизации (winrate) ---
#include "TargetWinRateCriterion.mqh"
//+------------------------------------------------------------------+
//| Глобальные переменные                                            |
//+------------------------------------------------------------------+
CTrade                  trade;

// Индикаторные хендлы: 0=EMA8, 1=EMA21, 2=RSI, 3=Stoch, 4=MACD, 5=ATR
int                     handles[6];

// Структура записи сделки (всегда не более одной активной)
struct DealRecord
  {
   ulong             ticket;
   int               type;              // 0=BUY, 1=SELL
   datetime          signal_time;
   int               held_bars;
   double            entry_price;
   double            sl;
   double            tp;
  };

DealRecord            deals[];
int                   deals_count = 0;

// Последние confidence обеих сетей (для строки состояния)
double                g_last_conf_buy  = 0.0;
double                g_last_conf_sell = 0.0;
string                g_last_decision  = "PASS";

// --- Графические панели (OBJ_LABEL): слева - мониторинг, справа - инфо о сделке ---
string                g_panel_left  = "RNNFilter_Monitor";
string                g_panel_right = "RNNFilter_Trade";
string                g_trade_info  = "Сделка: не открывалась";

// --- Перемещаемые панели (по статье mql5.com/ru/articles/12923) ---
// Торговая панель (кнопки Buy/Sell + лот) и панель логов
string                g_pan = "RNN_Panel";      // префикс торговой панели
string                g_logp= "RNN_Log";        // префикс панели логов
#define PANEL_OFFSET_X   190                    // горизонтальный сдвиг лог-панели
#define PANEL_OFFSET_Y   20
#define DRAG_TOLERANCE   3

// Координаты текущей позиции перемещения панелей
int                   g_panel_drag_x = 0;        // X заголовка при захвате (изменяемая)
int                   g_panel_drag_y = 0;
int                   g_panel_mid_x  = 0;
int                   g_panel_mid_y  = 0;

// Лог-буфер
string                g_log_lines[];

// --- Состояние перетаскивания панелей (мышь) ---
bool                  g_drag_pan  = true;
bool                  g_drag_log  = true;
int                   g_drag_dx   = 0;
int                   g_drag_dy   = 0;

int                   nn_buy_seq_len = 0;
int                   nn_sell_seq_len = 0;
datetime              last_bar_time = 0;

double                sl_price = 0.0;
double                trail_points = 0.0;
double                trail_start = 0.0;
double                be_points = 0.0;
double                be_offset = 0.0;
string                g_price_fmt = "%.2f";

// Буферы для индикаторов и рыночных данных
double                open_buf[], close_buf[], high_buf[], low_buf[];
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
   if(Inp_HoldBars < 1 || Inp_LotSize <= 0)
     {
      Print("Ошибка параметров: HoldBars>=1, LotSize>0");
      return(INIT_FAILED);
     }

   g_price_fmt = StringFormat("%%.%df", _Digits);

// --- Валидация порогов НС ---
   if(Inp_BuyConfThreshold <= 0.0 || Inp_SellConfThreshold <= 0.0)
     {
      Print("Ошибка порогов НС: пороги уверенности должны быть > 0");
      return(INIT_FAILED);
     }

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
         Print("RNNFilter: не удалось загрузить ONNX-модели. Файлы должны лежать в Common\\Files\\RNN_GOLD\\");
         return(INIT_FAILED);
        }
     }

   PrintFormat("RNNFilter инициализирован. NN: ONNX (%s | %s), пороги BUY=%.2f SELL=%.2f, RR: 1:%.1f, SL: %.1f п., Hold: %d",
               Inp_BuyModelFile, Inp_SellModelFile,
               Inp_BuyConfThreshold, Inp_SellConfThreshold,
               Inp_RR, Inp_SL_Points, Inp_HoldBars);
   UpdateStatus();


   UpdateStatus();
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Деинициализация                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
// Закрываем оставшиеся позиции
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
     {
      UpdateStatus();
      return;
     }

   datetime cur_bar = iTime(_Symbol, PERIOD_M15, 0);
   if(cur_bar == last_bar_time)
     {
      UpdateStatus();
      return;  // Та же самая свеча — ничего не делаем
     }

   last_bar_time = cur_bar;

// На последнем баре дня закрываем все позиции
   if(IsLastBarOfDay(cur_bar))
     {
      CloseAllDeals();
      UpdateStatus();
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
     {
      UpdateStatus();
      return;
     }
   if(Inp_NewsFilterEnable &&
      IsNewsBlocked(Inp_NewsEventTime, Inp_NewsWindowBefore, Inp_NewsWindowAfter))
     {
      UpdateStatus();
      return;
     }

// --- Проверяем сигналы ---
   CheckSignals();

// --- Строка состояния ---
   UpdateStatus();
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
//| Есть ли уже открытая сделка (только одна активная)               |
//+------------------------------------------------------------------+
bool HasOpenPosition()
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != Inp_Magic)
         continue;
      return(true);
     }
   return(false);
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
         if((bid - entry) < trail_start)
            continue;

         double new_sl = bid - trail_points;
         if(new_sl > pos_sl && new_sl > entry)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
            PrintFormat("TRAIL CHANGE: BUY #%I64u SL %.2f -> %.2f (bid=%.2f)", deals[i].ticket, pos_sl, new_sl, bid);
           }
        }
      else // SELL
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if((entry - ask) < trail_start)
            continue;

         double new_sl = ask + trail_points;
         if((new_sl < pos_sl || pos_sl == 0) && new_sl < entry)
           {
            double tp = (deals[i].tp > 0) ? deals[i].tp : 0;
            if(trade.PositionModify(deals[i].ticket, new_sl, tp))
               deals[i].sl = new_sl;
            PrintFormat("TRAIL CHANGE: SELL #%I64u SL %.2f -> %.2f (ask=%.2f)", deals[i].ticket, pos_sl, new_sl, ask);
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
         if((bid - entry) < be_points)
            continue;

         double new_sl = entry + be_offset;
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
         if((entry - ask) < be_points)
            continue;

         double new_sl = entry - be_offset;
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

// Копируем по максимуму, но ДОПУСКАЕМ меньше истории (накопление)
   int avail = CopyClose(_Symbol, PERIOD_M15, 0, cnt, close_buf);
   if(avail < min_need)
      return;
   int n;
   n = CopyOpen(_Symbol, PERIOD_M15, 0, cnt, open_buf);
   if(n < avail)
      avail = n;
   n = CopyHigh(_Symbol, PERIOD_M15, 0, cnt, high_buf);
   if(n < avail)
      avail = n;
   n = CopyLow(_Symbol, PERIOD_M15, 0, cnt, low_buf);
   if(n < avail)
      avail = n;
   n = CopyTickVolume(_Symbol, PERIOD_M15, 0, cnt, volume_buf);
   if(n < avail)
      avail = n;

   n = CopyBuffer(handles[0], 0, 0, cnt, ema8_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[1], 0, 0, cnt, ema21_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[2], 0, 0, cnt, rsi_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[3], 0, 0, cnt, stochK_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[3], 1, 0, cnt, stochD_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[4], 0, 0, cnt, macd_main_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[4], 1, 0, cnt, macd_signal_buf);
   if(n < avail)
      avail = n;
   n = CopyBuffer(handles[5], 0, 0, cnt, atr_buf);
   if(n < avail)
      avail = n;

   if(avail < min_need)
      return;
   int avail_bars = avail;

   const int shift = 1;  // последний закрытый бар
   datetime signal_time = iTime(_Symbol, PERIOD_M15, shift);
   if(IsLastBarOfDay(signal_time))
      return;  // сигналы на последнем баре дня не открываем

// --- Детекция сырых сигналов ---
   bool buy_signal = false, sell_signal = false;

   if(ema8_buf[shift] > ema21_buf[shift] && ema8_buf[shift + 1] <= ema21_buf[shift + 1])
      buy_signal = true;
   if(ema8_buf[shift] < ema21_buf[shift] && ema8_buf[shift + 1] >= ema21_buf[shift + 1])
      sell_signal = true;

   if(rsi_buf[shift] > 38 && rsi_buf[shift + 1] <= 38)
      buy_signal = true;
   if(rsi_buf[shift] < 62 && rsi_buf[shift + 1] >= 62)
      sell_signal = true;

   if(stochK_buf[shift] > stochD_buf[shift] &&
      stochK_buf[shift + 1] <= stochD_buf[shift + 1] && stochK_buf[shift] <= 30)
      buy_signal = true;
   if(stochK_buf[shift] < stochD_buf[shift] &&
      stochK_buf[shift + 1] >= stochD_buf[shift + 1] && stochK_buf[shift] >= 70)
      sell_signal = true;

   if(macd_main_buf[shift] > macd_signal_buf[shift] &&
      macd_main_buf[shift + 1] <= macd_signal_buf[shift + 1])
      buy_signal = true;
   if(macd_main_buf[shift] < macd_signal_buf[shift] &&
      macd_main_buf[shift + 1] >= macd_signal_buf[shift + 1])
      sell_signal = true;

// --- Прогон ОБЕИХ сетей ONNX (для решения и строки состояния) ---
   double conf_buy = 0.0, conf_sell = 0.0;
   double p0_buy = 0.0, p1_buy = 0.0, p0_sell = 0.0, p1_sell = 0.0;
   bool   nn_active = (Inp_UseNNFilter && NN_ONNX_IsReady());

   if(nn_active)
     {
      double buy_features[], sell_features[];
      ArrayResize(buy_features, NN_BUY_NORM_WINDOW * NN_FEATURES);
      NN_BuildFeatures(shift, open_buf, close_buf, high_buf, low_buf, volume_buf,
                       ema8_buf, ema21_buf, rsi_buf,
                       stochK_buf, stochD_buf,
                       macd_main_buf, macd_signal_buf, atr_buf,
                       NN_BUY_NORM_WINDOW, avail_bars, buy_features);
      conf_buy = NN_OnnxRunModel(0, buy_features, p0_buy, p1_buy) ? ((Inp_NNProfitClass == 0) ? p0_buy : p1_buy) : -1.0;

      ArrayResize(sell_features, NN_SELL_NORM_WINDOW * NN_FEATURES);
      NN_BuildFeatures(shift, open_buf, close_buf, high_buf, low_buf, volume_buf,
                       ema8_buf, ema21_buf, rsi_buf,
                       stochK_buf, stochD_buf,
                       macd_main_buf, macd_signal_buf, atr_buf,
                       NN_SELL_NORM_WINDOW, avail_bars, sell_features);
      conf_sell = NN_OnnxRunModel(1, sell_features, p0_sell, p1_sell) ? ((Inp_NNProfitClass == 0) ? p0_sell : p1_sell) : -1.0;

      if(conf_buy < 0.0 || conf_sell < 0.0)
        {
         PrintFormat("RNNFilter: ошибка ONNX-инференса (BUY=%.2f SELL=%.2f)", conf_buy, conf_sell);
         return;
        }
     }

   g_last_conf_buy  = conf_buy;
   g_last_conf_sell = conf_sell;

// --- Решение: одна сеть уверена в покупке, другая НЕ уверена в продаже ---
   bool buy_ok = false, sell_ok = false;

   if(!Inp_TradeWOIndicators && buy_signal && Inp_BuySignal && !HasOpenPosition())
     {
      if(!nn_active)
         buy_ok = true;
      else
         if(conf_buy >= Inp_BuyConfThreshold && conf_sell < Inp_SellConfThreshold)
            buy_ok = true;
     }

   if(!Inp_TradeWOIndicators && sell_signal && Inp_SellSignal && !HasOpenPosition())
     {
      if(!nn_active)
         sell_ok = true;
      else
         if(conf_sell >= Inp_SellConfThreshold && conf_buy < Inp_BuyConfThreshold)
            sell_ok = true;
     }

// --- Режим БЕЗ индикаторных сигналов: только по доверию сетей ---
   if(Inp_TradeWOIndicators && !HasOpenPosition() && nn_active)
     {
      // BUY : BUY-сеть уверена в покупке (class1 >= порог BUY)
      //       И SELL-сеть уверена, что продавать НЕ надо (class0 >= порог SELL).
      // SELL: симметрично.
      double buy_class0  = 1.0 - conf_buy;    // softmax: P0 + P1 = 1
      double sell_class0 = 1.0 - conf_sell;

      if(Inp_BuySignal && conf_buy >= Inp_BuyConfThreshold &&
         sell_class0 >= Inp_SellConfThreshold)
         buy_ok = true;

      if(Inp_SellSignal && conf_sell >= Inp_SellConfThreshold &&
         buy_class0 >= Inp_BuyConfThreshold)
         sell_ok = true;
     }

   g_last_decision = (buy_ok ? "BUY" : (sell_ok ? "SELL" : "PASS"));

// --- Лог предсказаний обеих сетей на КАЖДОМ баре ---
   if(nn_active)
     {


      PrintFormat("Итог %s confidence [%.4f, %.4f] [%.4f, %.4f]",
                  (buy_ok ? "BUY" : (sell_ok ? "SELL" : "IDLE")),
                  p0_buy, p1_buy, p0_sell, p1_sell);




     }

// Открываем максимум одну сделку за бар
   if(buy_ok)
     {
      ProcessBuySignal(signal_time, conf_buy, conf_sell);
      return;
     }
   if(sell_ok)
     {
      ProcessSellSignal(signal_time, conf_sell, conf_buy);
      return;
     }
  }
//+------------------------------------------------------------------+
//| Обработка BUY сигнала (проверка НС -> исполнение)                |
//+------------------------------------------------------------------+
void ProcessBuySignal(datetime signal_time, double conf_buy, double conf_sell)
  {
   if(Inp_UseNNFilter)
     {
      // BUY разрешён, если BUY-сеть уверена в покупке И SELL-сеть НЕ уверена в продаже
      if(!(conf_buy >= Inp_BuyConfThreshold && conf_sell < Inp_SellConfThreshold))
        {
         PrintFormat("BUY отклонён: BUYconf=%.4f (>=%.2f) SELLconf=%.4f (<%.2f)",
                     conf_buy, Inp_BuyConfThreshold, conf_sell, Inp_SellConfThreshold);
         return;
        }
     }

   ExecuteBuySignal(signal_time, conf_buy, conf_sell);
  }
//+------------------------------------------------------------------+
//| Исполнение BUY (только одна активная сделка, фикс. лот)          |
//+------------------------------------------------------------------+
void ExecuteBuySignal(datetime signal_time, double conf_buy, double conf_sell)
  {
// Только одна активная сделка
   if(HasOpenPosition())
      return;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double sl  = (Inp_SL_Points > 0.0) ? ask - sl_price : 0.0;
   double tp  = (sl > 0.0 && Inp_RR > 0.0) ? ask + Inp_RR * (ask - sl) : 0.0;
   double lot = NormalizeLot(Inp_LotSize);

   string cmt = "RNN_Buy";
   if(Inp_UseNNFilter)
      cmt = StringFormat("B%.2f/S%.2f", conf_buy, conf_sell);

   if(trade.Buy(lot, _Symbol, ask, sl, tp, cmt))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, 0, signal_time, ask, sl, tp);
         g_trade_info = StringFormat("СДЕЛКА BUY  #%I64u\nЛот: %.2f\nЦена: " + g_price_fmt +
                                     "\nSL: " + g_price_fmt + "\nTP: " + g_price_fmt +
                                     "\nBUYconf = %.2f  SELLconf = %.2f",
                                     ticket, lot, ask, sl, tp, conf_buy, conf_sell);
         PrintFormat("BUY #%I64u | Lot %.2f | Ask " + g_price_fmt + " | SL " + g_price_fmt + " | TP " + g_price_fmt + " | BUYconf=%.2f SELLconf=%.2f",
                     ticket, lot, ask, sl, tp, conf_buy, conf_sell);
        }
     }
  }
//+------------------------------------------------------------------+
//| Обработка SELL сигнала (проверка НС -> исполнение)               |
//+------------------------------------------------------------------+
void ProcessSellSignal(datetime signal_time, double conf_sell, double conf_buy)
  {
   if(Inp_UseNNFilter)
     {
      // SELL разрешён, если SELL-сеть уверена в продаже И BUY-сеть НЕ уверена в покупке
      if(!(conf_sell >= Inp_SellConfThreshold && conf_buy < Inp_BuyConfThreshold))
        {
         PrintFormat("SELL отклонён: SELLconf=%.4f (>=%.2f) BUYconf=%.4f (<%.2f)",
                     conf_sell, Inp_SellConfThreshold, conf_buy, Inp_BuyConfThreshold);
         return;
        }
     }

   ExecuteSellSignal(signal_time, conf_sell, conf_buy);
  }
//+------------------------------------------------------------------+
//| Исполнение SELL (только одна активная сделка, фикс. лот)         |
//+------------------------------------------------------------------+
void ExecuteSellSignal(datetime signal_time, double conf_sell, double conf_buy)
  {
// Только одна активная сделка
   if(HasOpenPosition())
      return;

   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl  = (Inp_SL_Points > 0.0) ? bid + sl_price : 0.0;
   double tp  = (sl > 0.0 && Inp_RR > 0.0) ? bid - Inp_RR * (sl - bid) : 0.0;
   double lot = NormalizeLot(Inp_LotSize);

   string cmt = "RNN_Sell";
   if(Inp_UseNNFilter)
      cmt = StringFormat("S%.2f/B%.2f", conf_sell, conf_buy);

   if(trade.Sell(lot, _Symbol, bid, sl, tp, cmt))
     {
      ulong ticket = trade.ResultOrder();
      if(ticket > 0)
        {
         AddDeal(ticket, 1, signal_time, bid, sl, tp);
         g_trade_info = StringFormat("СДЕЛКА SELL  #%I64u\nЛот: %.2f\nЦена: " + g_price_fmt +
                                     "\nSL: " + g_price_fmt + "\nTP: " + g_price_fmt +
                                     "\nSELLconf = %.2f  BUYconf = %.2f",
                                     ticket, lot, bid, sl, tp, conf_sell, conf_buy);
         PrintFormat("SELL #%I64u | Lot %.2f | Bid " + g_price_fmt + " | SL " + g_price_fmt + " | TP " + g_price_fmt + " | SELLconf=%.2f BUYconf=%.2f",
                     ticket, lot, bid, sl, tp, conf_sell, conf_buy);
        }
     }
  }
//+------------------------------------------------------------------+
//| Добавление сделки в список отслеживания                          |
//+------------------------------------------------------------------+
void AddDeal(ulong ticket, int type, datetime signal_time, double entry, double sl, double tp)
  {
   int idx = deals_count++;
   ArrayResize(deals, deals_count);
   deals[idx].ticket      = ticket;
   deals[idx].type        = type;
   deals[idx].signal_time = signal_time;
   deals[idx].held_bars   = 0;
   deals[idx].entry_price = entry;
   deals[idx].sl          = sl;
   deals[idx].tp          = tp;
  }
//+------------------------------------------------------------------+
//| Строка состояния: confidence обеих сетей + открытая сделка       |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Настройка/обновление текстовой метки-панели                      |
//+------------------------------------------------------------------+
void Panel_Render(const string name, const int corner, const int xd, const int yd,
                  const string text, const color txtclr, const color bgclr)
  {
// --- Серый фон-подложка ---
   string bg = name + "_BG";
   if(ObjectFind(0, bg) < 0)
     {
      if(!ObjectCreate(0, bg, OBJ_RECTANGLE_LABEL, 0, 0, 0))
         return;
      ObjectSetInteger(0, bg, OBJPROP_CORNER, corner);
      ObjectSetInteger(0, bg, OBJPROP_XDISTANCE, xd);
      ObjectSetInteger(0, bg, OBJPROP_YDISTANCE, yd);
      ObjectSetInteger(0, bg, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, bg, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, bg, OBJPROP_ZORDER, 0);
      ObjectSetInteger(0, bg, OBJPROP_BACK, true);   // на задний план, чтобы не перекрывать текст
      ObjectSetInteger(0, bg, OBJPROP_BGCOLOR, bgclr);
      ObjectSetInteger(0, bg, OBJPROP_FILL, true);
      ObjectSetInteger(0, bg, OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, bg, OBJPROP_WIDTH, 1);
     }

// --- Размер фона по тексту ---
   string rows[];
   int n = StringSplit(text, '\n', rows);
   if(n < 1)
      n = 1;
   int maxlen = 0;
   for(int i = 0; i < n; i++)
      maxlen = MathMax(maxlen, StringLen(rows[i]));
   int fs = 10;
   ObjectSetInteger(0, bg, OBJPROP_XSIZE, maxlen * (fs / 2) + 14);
   ObjectSetInteger(0, bg, OBJPROP_YSIZE, n * (fs + 2) + 8);

// --- Текст поверх фона ---
   if(ObjectFind(0, name) < 0)
     {
      if(!ObjectCreate(0, name, OBJ_LABEL, 0, 0, 0))
         return;
      ObjectSetInteger(0, name, OBJPROP_CORNER, corner);
      ObjectSetInteger(0, name, OBJPROP_XDISTANCE, xd + 7);
      ObjectSetInteger(0, name, OBJPROP_YDISTANCE, yd + 4);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
      ObjectSetInteger(0, name, OBJPROP_ZORDER, 1);
      ObjectSetString(0, name, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fs);
     }
   ObjectSetInteger(0, name, OBJPROP_COLOR, txtclr);
   ObjectSetString(0, name, OBJPROP_TEXT, text);
  }
//+------------------------------------------------------------------+
//| Удаление метки-панели (текст + фон)                              |
//+------------------------------------------------------------------+
void Panel_Delete(const string &name)
  {
   string bg = name + "_BG";
   if(ObjectFind(0, bg) >= 0)
      ObjectDelete(0, bg);
   if(ObjectFind(0, name) >= 0)
      ObjectDelete(0, name);
  }
//+==================================================================+
//|  Перемещаемые панели (по статье mql5.com/ru/articles/12923)      |
//|  Панель 1 — торговая (кнопки BUY/SELL + лот).                    |
//|  Панель 2 — список всех логов советника.                         |
//+------------------------------------------------------------------+
int    g_ptx = 20;                          // базовая X торговой панели
int    g_pty = 80;                          // базовая Y торговой панели
int    g_plx = 0;                           // базовая X лог-панели
int    g_ply = 0;                           // базовая Y лог-панели
double g_pnl_lot = 0.0;                     // рабочий лот для кнопок

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void   LogAdd(string msg);                  // вперёд-объявление
void   ExecuteOpenManualTrade(ENUM_ORDER_TYPE typ);   // вперёд-объявление

//+------------------------------------------------------------------+
//| Создать/обновить прямоугольную метку (подложку)                  |
//+------------------------------------------------------------------+
void Pn_Rect(const string name, const int x, const int y, const int w, const int h,
             const color bg)
  {
   if(ObjectFind(0,name) < 0)
     {
      ObjectCreate(0,name,OBJ_RECTANGLE_LABEL,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,0);
      ObjectSetInteger(0,name,OBJPROP_FILL,true);
      ObjectSetInteger(0,name,OBJPROP_BORDER_TYPE,BORDER_FLAT);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
   ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bg);
  }
//+------------------------------------------------------------------+
//| Создать/обновить текстовую метку                                 |
//+------------------------------------------------------------------+
void Pn_Lbl(const string name, const int x, const int y, const string txt,
            const color c, const int fs=10)
  {
   if(ObjectFind(0,name)<0)
     {
      ObjectCreate(0,name,OBJ_LABEL,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,false);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,true);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,2);
      ObjectSetString(0,name,OBJPROP_FONT,"Consolas");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,fs);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_COLOR,c);
   ObjectSetString(0,name,OBJPROP_TEXT,txt);
  }
//+------------------------------------------------------------------+
//| Создать/обновить кнопку                                          |
//+------------------------------------------------------------------+
void Pn_Btn(const string name, const int x, const int y, const int w, const int h,
            const string txt, const color bg, const color fg)
  {
   if(ObjectFind(0,name)<0)
     {
      ObjectCreate(0,name,OBJ_BUTTON,0,0,0);
      ObjectSetInteger(0,name,OBJPROP_CORNER,CORNER_LEFT_UPPER);
      ObjectSetInteger(0,name,OBJPROP_SELECTABLE,true);
      ObjectSetInteger(0,name,OBJPROP_HIDDEN,false);
      ObjectSetInteger(0,name,OBJPROP_ZORDER,0);
      ObjectSetString(0,name,OBJPROP_FONT,"Arial");
      ObjectSetInteger(0,name,OBJPROP_FONTSIZE,11);
      ObjectSetInteger(0,name,OBJPROP_XSIZE,w);
      ObjectSetInteger(0,name,OBJPROP_YSIZE,h);
     }
   ObjectSetInteger(0,name,OBJPROP_XDISTANCE,x);
   ObjectSetInteger(0,name,OBJPROP_YDISTANCE,y);
   ObjectSetInteger(0,name,OBJPROP_BGCOLOR,bg);
   ObjectSetInteger(0,name,OBJPROP_COLOR,fg);
   ObjectSetString(0,name,OBJPROP_TEXT,txt);
  }
//+------------------------------------------------------------------+
//| Сборка перемещаемых панелей                                      |
//+------------------------------------------------------------------+
//| Сборка перемещаемых панелей                                      |
//+------------------------------------------------------------------+
void Panel_Build()
  {
// Торговая панель (слева сверху)
   g_ptx=20;
   g_pty=60;
   Pn_Rect(g_pan+"_Title", g_ptx, g_pty, 190, 26, C'47,54,153');
   Pn_Lbl(g_pan+"_TitleTxt", g_ptx+8, g_pty+5, "RNF Filter Trading", clrWhite, 11);
   Pn_Rect(g_pan+"_Border", g_ptx+2, g_pty+27, 186, 62, C'90,90,96');
   Pn_Lbl(g_pan+"_LotTxt", g_ptx+12, g_pty+36, "Lot:", clrSilver, 11);
   g_pnl_lot=Inp_LotSize;
   Pn_Lbl(g_pan+"_LotVal", g_ptx+46, g_pty+34,
          DoubleToString(NormalizeLot(g_pnl_lot),2), clrWhite, 12);
   Pn_Btn(g_pan+"_Buy", g_ptx+12, g_pty+64, 78, 28, "BUY", C'0,128,0', clrWhite);
   Pn_Btn(g_pan+"_Sell", g_ptx+104, g_pty+64, 78, 28, "SELL", C'180,0,0', clrWhite);

// Лог-панель (справа)
   g_plx=310;
   g_ply=60;
   Pn_Rect(g_logp+"_Title", g_plx, g_ply, 340, 26, C'64,64,80');
   Pn_Lbl(g_logp+"_TitleTxt", g_plx+8, g_ply+3, "RNNFilter Logs", clrWhite, 11);
   Pn_Rect(g_logp+"_Border", g_plx+2, g_ply+27, 336, 260, C'70,70,76');
   Pn_Lbl(g_logp+"_Body", g_plx+8, g_ply+32, "Лог пуст", clrSilver, 9);

// Заголовки делаем перетаскиваемыми (управляющий элемент)
   ObjectSetInteger(0,g_pan+"_Title",OBJPROP_SELECTABLE,true);
   ObjectSetInteger(0,g_pan+"_Title",OBJPROP_HIDDEN,false);
   ObjectSetInteger(0,g_logp+"_Title",OBJPROP_SELECTABLE,true);
   ObjectSetInteger(0,g_logp+"_Title",OBJPROP_HIDDEN,false);
   ChartRedraw(0);
  }
//+------------------------------------------------------------------+
//| Уничтожение графических объектов панелей                         |
//+------------------------------------------------------------------+
void Panel_Destroy()
  {
   string del[] = {"_Title","_TitleTxt","_Border","_LotTxt","_LotVal","_Buy","_Sell"};
   for(int i=0;i<ArraySize(del);i++)
     {
      DeleteObj(g_pan+del[i]);
     }
   for(int i=0;i<ArraySize(del);i++)
     {
      DeleteObj(g_logp+del[i]);
     }
   DeleteObj(g_logp+"_Body");
  }
//+------------------------------------------------------------------+
//| Удалить объект по имени (обёртка)                              |
//+------------------------------------------------------------------+
void DeleteObj(const string nm)
  {
   if(ObjectFind(0,nm)>=0)
      ObjectDelete(0,nm);
  }
//+------------------------------------------------------------------+
//| Добавить запись в лог и обновить лог-панель                      |
//+------------------------------------------------------------------+
void LogAdd(string msg)
  {
// Логи пишем только в журнал терминала, на график не выводим
   Print(msg);
  }
//+------------------------------------------------------------------+
//| Обновить информационный блок панели RNN_Log                      |
//| 1: пороги сетей (из параметров)                                  |
//| 2: confidence BUY | 3: confidence SELL                           |
//| 4: решение PASS/BUY/SELL (просто информация)                     |
//+------------------------------------------------------------------+
void Panel_LogInfo()
  {
   string buy_txt  = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_buy)
                     : "0.0000 (NN off)";
   string sell_txt = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_sell)
                     : "0.0000 (NN off)";
   string info = StringFormat("Пороги   BUY=%.2f  SELL=%.2f", Inp_BuyConfThreshold, Inp_SellConfThreshold) + "\n"
                 + StringFormat("Conf BUY = %s", buy_txt) + "\n"
                 + StringFormat("Conf SELL = %s", sell_txt) + "\n"
                 + StringFormat("Решение: %s", g_last_decision);
   Pn_Lbl(g_logp+"_Body", g_plx+8, g_ply+32, info, clrWhite, 11);
  }
//+------------------------------------------------------------------+
//| Обработка кликов и перетаскивания панелей                        |
//+------------------------------------------------------------------+
void OnChartEvent(const int id,const long &lparam,const double &dparam,const string &sparam)
  {
// --- Перетаскивание панелей мышью (Rectangle Label не тянется нативно) ---
   if(id==CHARTEVENT_MOUSE_MOVE)
     {
      int mx=(int)lparam;
      int my=(int)dparam;
      bool left=((StringToInteger(sparam) & 1)!=0);   // левая кнопка нажата

      if(!left)
        {
         g_drag_pan=false;
         g_drag_log=false;
         return;
        }

      if(!g_drag_pan && !g_drag_log)
        {
         // начало перетаскивания: курсор в заголовной полосе (190x26)
         if(mx>=g_ptx && mx<=g_ptx+190 && my>=g_pty && my<=g_pty+26)
           { g_drag_pan=true; g_drag_dx=mx-g_ptx; g_drag_dy=my-g_pty; }
         else
            if(mx>=g_plx && mx<=g_plx+340 && my>=g_ply && my<=g_ply+26)
              { g_drag_log=true; g_drag_dx=mx-g_plx; g_drag_dy=my-g_ply; }
        }

      if(g_drag_pan)
        {
         int nx=mx-g_drag_dx, ny=my-g_drag_dy;
         ShiftPanel(g_pan, nx-g_ptx, ny-g_pty);
         g_ptx=nx;
         g_pty=ny;
         ChartRedraw(0);
        }
      else
         if(g_drag_log)
           {
            int nx=mx-g_drag_dx, ny=my-g_drag_dy;
            ShiftPanel(g_logp, nx-g_plx, ny-g_ply);
            g_plx=nx;
            g_ply=ny;
            ChartRedraw(0);
           }
     }
// --- Клики по кнопкам BUY/SELL ---
   if(id==CHARTEVENT_OBJECT_CLICK)
     {
      if(sparam==g_pan+"_Buy")
         ExecuteOpenManualTrade(ORDER_TYPE_BUY);
      else
         if(sparam==g_pan+"_Sell")
            ExecuteOpenManualTrade(ORDER_TYPE_SELL);
     }
  }
//+------------------------------------------------------------------+
//| Сдвинуть все объекты панели с общим префиксом на dx,dy          |
//+------------------------------------------------------------------+
void ShiftPanel(const string prefix,const int dx,const int dy)
  {
   string parts[] = {"_Title","_TitleTxt","_Border","_LotTxt","_LotVal","_Buy","_Sell","_Body"};
   for(int i=0;i<ArraySize(parts);i++)
     {
      string nm=prefix+parts[i];
      if(ObjectFind(0,nm)<0)
         continue;
      int x=(int)ObjectGetInteger(0,nm,OBJPROP_XDISTANCE)+dx;
      int y=(int)ObjectGetInteger(0,nm,OBJPROP_YDISTANCE)+dy;
      ObjectSetInteger(0,nm,OBJPROP_XDISTANCE,x);
      ObjectSetInteger(0,nm,OBJPROP_YDISTANCE,y);
     }
  }
//+------------------------------------------------------------------+
//| Ручное открытие рыночной сделки по кнопке панели                |
//+------------------------------------------------------------------+
void ExecuteOpenManualTrade(const ENUM_ORDER_TYPE typ)
  {
   if(HasOpenPosition())
     {
      LogAdd("Ручной вход: уже есть открытая позиция");
      return;
     }
   if(!Inp_Enabled)
     {
      LogAdd("Ручной вход заблокирован");
      return;
     }
   double lot=NormalizeLot(g_pnl_lot);
   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid=SymbolInfoDouble(_Symbol,SYMBOL_BID);
   double sl=0.0, tprice=0.0;
   if(Inp_SL_Points>0)
     {
      if(typ==ORDER_TYPE_BUY)
         sl=ask-Inp_SL_Points*_Point;
      else
         sl=bid+Inp_SL_Points*_Point;
      if(Inp_RR>0)
        {
         if(typ==ORDER_TYPE_BUY)
            tprice=ask+Inp_RR*(ask-sl);
         else
            tprice=bid-Inp_RR*(sl-bid);
        }
     }
   bool ok=false;
   if(typ==ORDER_TYPE_BUY)
      ok=trade.Buy(lot,_Symbol,ask,sl,tprice,"RNN_WO_Man");
   else
      if(typ==ORDER_TYPE_SELL)
         ok=trade.Sell(lot,_Symbol,bid,sl,tprice,"RNN_WO_Man");
   if(ok)
     {
      ulong ticket=trade.ResultOrder();
      LogAdd(StringFormat("Ручная сделка %s #%I64u лот %.2f",
                          (typ==ORDER_TYPE_BUY)?"BUY":"SELL",ticket,lot));
      // добавить в отслеживание как обычную сделку без НС
      AddDeal(ticket, (typ==ORDER_TYPE_BUY)?0:1, TimeCurrent(),
              (typ==ORDER_TYPE_BUY)?ask:bid, sl, tprice);
     }
   else
      LogAdd(StringFormat("Ошибка ручной сделки: %d",GetLastError()));
   UpdateStatus();
  }
//+------------------------------------------------------------------+
//| Строка состояния (panels)                                        |
//+------------------------------------------------------------------+
void UpdateStatus()
  {
   string conf_buy  = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_buy)
                      : StringFormat("%.4f (NN off)", 0.0);
   string conf_sell = (Inp_UseNNFilter && NN_ONNX_IsReady()) ? StringFormat("%.4f", g_last_conf_sell)
                      : StringFormat("%.4f (NN off)", 0.0);

   string pos = "Сделок: нет";
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong t = PositionGetTicket(i);
      if(t == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != Inp_Magic)
         continue;
      string stype = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
      double vol    = PositionGetDouble(POSITION_VOLUME);
      double profit = PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
      pos = StringFormat("Сделка: %s  vol=%.2f  profit=%.2f", stype, vol, profit);
      break;
     }

   string mode_txt = (Inp_TradeWOIndicators) ?
                     "Режим: БЕЗ индикаторов - только по доверию сетей" :
                     "Режим: индикаторные сигналы + подтверждение сетей";
   string monitor = "RNNFilter   (одна активная сделка)\n"
                    + StringFormat("BUY  сеть:  conf = %s   (порог %.2f)\n", conf_buy, Inp_BuyConfThreshold)
                    + StringFormat("SELL сеть:  conf = %s   (порог %.2f)\n", conf_sell, Inp_SellConfThreshold)
                    + mode_txt + "\n"
                    + "Условие BUY: BUY-сеть уверена И SELL-сеть НЕ уверена (и наоборот)\n"
                    + pos;

// Информативный текст: мониторинг (слева) на сером фоне


// ПРАВАЯ панель: блок информации о сделке (открытие) на сером фоне


// Панель RNN_Log: пороги + confidence + решение (информация)






  }
//+------------------------------------------------------------------+
