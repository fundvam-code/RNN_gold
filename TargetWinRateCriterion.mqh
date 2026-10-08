//+------------------------------------------------------------------+
//|                                        TargetWinRateCriterion.mqh |
//|   Пользовательский критерий оптимизации: WINRATE (% прибыльных)  |
//|   Версия 2.0 - совместима со скриптом CheckOptimization.mq5      |
//|                                                                  |
//|   Назначение:                                                    |
//|     Отдельный подключаемый модуль, который определяет функцию    |
//|     OnTester() для советника. При оптимизации в тестере          |
//|     выберите критерий "Пользовательский максимум" (Custom max),  |
//|     и тестер будет подбирать параметры по winrate, а не по       |
//|     балансу/профит-фактору.                                      |
//|                                                                  |
//|   Что нового в v2:                                               |
//|     1) Кадр FrameAdd начинается с маркера WR_MAGIC (12345.6789); |
//|        далее 11 метрик: winrate общий/BUY/SELL, Profit Factor,   |
//|        Z-счёт, просадка баланса/equity, сделки, победы,          |
//|        поражения, P/L. Эту раскладку ожидает CheckOptimization.  |
//|     2) CSV-выгрузка по каждому проходу осталась (Common\Files\   |
//|        TargetWinRate_report.csv).                                |
//|                                                                  |
//|   Подключение (ВАЖНО):                                           |
//|     #include "TargetWinRateCriterion.mqh"                        |
//|     Должен идти ПОСЛЕ блока input-параметров, т.к. модуль        |
//|     использует входной параметр Inp_Magic для фильтра сделок.    |
//|                                                                  |
//|   Считается: количество закрытых позиций с прибылью / общее      |
//|   число закрытых позиций (по сделкам DEAL_ENTRY_OUT/INOUT).      |
//|   Возвращается в процентах (0..100).                             |
//+------------------------------------------------------------------+
#ifndef TARGET_WINRATE_CRITERION_MQH
#define TARGET_WINRATE_CRITERION_MQH

//--- Конфигурация критерия (при необходимости отредактируйте) ---
#define WR_USE_MAGIC_FILTER      // учитывать только сделки с Inp_Magic
#define WR_USE_SYMBOL_FILTER     // учитывать только текущий символ
#define WR_RETURN_PERCENT        // возвращать % (0..100); иначе долю (0..1)
#define WR_FRAME_REPORT          // кадр FrameAdd с метриками (для .mqd)
#define WR_CSV_DUMP              // CSV-отчёт по проходам оптимизации

//--- Маркер данных кадра: его ищет CheckOptimization.mq5 (InpMagicValue) ---
#define WR_MAGIC         12345.6789

#ifdef WR_CSV_DUMP
#define WR_CSV_FILENAME   "TargetWinRate_report.csv"   // в Common\Files\
int    g_wr_csv_handle = INVALID_HANDLE;
#endif

//+------------------------------------------------------------------+
//| Структура кадра (все поля double, чтобы FrameAdd сериализовал   |
//| без выравнивания; слова 0..11 читает CheckOptimization.mq5):    |
//|   0  = WR_MAGIC (маркер)                                         |
//|   1  = overall_wr    (общий winrate, %)                          |
//|   2  = buy_wr        (winrate BUY, %)                            |
//|   3  = sell_wr       (winrate SELL, %)                           |
//|   4  = profit_factor (Profit Factor)                             |
//|   5  = zscore        (Z-счёт серий)                              |
//|   6  = dd_balance_pct (просадка баланса, % от пика)              |
//|   7  = dd_equity_pct  (просадка equity, % - оценка)              |
//|   8  = trades        (всего закрытых позиций)                    |
//|   9  = wins          (прибыльных)                                |
//|   10 = losses        (убыточных)                                 |
//|   11 = profit_sum    (суммарный P/L)                             |
//|   12..15 = разбивка BUY/SELL (trades_buy/sell, wins_buy/sell)    |
//+------------------------------------------------------------------+
struct TargetWinRateFrame
  {
   double magic;
   double overall_wr;
   double buy_wr;
   double sell_wr;
   double profit_factor;
   double zscore;
   double dd_balance_pct;
   double dd_equity_pct;
   double trades;
   double wins;
   double losses;
   double profit_sum;
   double trades_buy;
   double trades_sell;
   double wins_buy;
   double wins_sell;
  };
//+------------------------------------------------------------------+
//| Сбор статистики и расчёт winrate/PF/Z/DD                         |
//+------------------------------------------------------------------+
double TargetWinRate_Collect(TargetWinRateFrame &frame)
  {
   // --- Обнуляем структуру и ставим маркер ---
   ZeroMemory(frame);
   frame.magic = WR_MAGIC;

   // В тестере история сделок доступна за весь период теста
   if(!HistorySelect(0, TimeCurrent()))
     {
      PrintFormat("TargetWinRate: HistorySelect() не удался (код %d)", GetLastError());
      return(0.0);
     }

   int total = HistoryDealsTotal();

   double gross_profit = 0.0;
   double gross_loss  = 0.0;
   double cum = 0.0, peak = 0.0, max_dd = 0.0;

   // Последовательность результатов (+1 победа, -1 поражение) для Z-счёта
   int res[];
   ArrayResize(res, 0);

   for(int i = 0; i < total; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0)
         continue;

      // Учитываем только позиционные сделки (buy/sell),
      // пропускаем балансовые операции, свопы, комиссии и т.п.
      long deal_type = HistoryDealGetInteger(ticket, DEAL_TYPE);
      if(deal_type != DEAL_TYPE_BUY && deal_type != DEAL_TYPE_SELL)
         continue;

      // Считаем winrate по закрывающим сделкам (выход из позиции),
      // чтобы одна закрытая позиция = один результат.
      long entry = HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_INOUT)
         continue;

#ifdef WR_USE_MAGIC_FILTER
      long magic = HistoryDealGetInteger(ticket, DEAL_MAGIC);
      if(magic != (long)Inp_Magic)
         continue;
#endif

#ifdef WR_USE_SYMBOL_FILTER
      string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;
#endif

      double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
      frame.profit_sum += profit;
      frame.trades += 1.0;

      // Накопленная кривая P/L для просадки
      cum += profit;
      if(cum > peak) peak = cum;
      double dd = peak - cum;
      if(dd > max_dd) max_dd = dd;

      // Разбивка BUY/SELL + серии для Z-счёта
      int outcome = 0; // 0 = ничья (в сериях не участвует)
      if(profit > 0.0)
        {
         frame.wins++;
         gross_profit += profit;
         outcome = 1;
        }
      else if(profit < 0.0)
        {
         frame.losses++;
         gross_loss -= profit;
         outcome = -1;
        }

      if(deal_type == DEAL_TYPE_BUY)
        {
         frame.trades_buy++;
         if(outcome == 1) frame.wins_buy++;
        }
      else // DEAL_TYPE_SELL
        {
         frame.trades_sell++;
         if(outcome == 1) frame.wins_sell++;
        }

      if(outcome != 0)
        {
         int n = ArraySize(res);
         ArrayResize(res, n + 1);
         res[n] = outcome;
        }
     }

   if(frame.trades == 0.0)
     {
      PrintFormat("TargetWinRate: нет закрытых сделок для критерия (magic=%d)", Inp_Magic);
      return(0.0);
     }

   // --- Winrate в долях ---
   double wr_all  = frame.wins / frame.trades;
   double wr_buy  = (frame.trades_buy  > 0) ? frame.wins_buy  / frame.trades_buy  : 0.0;
   double wr_sell = (frame.trades_sell > 0) ? frame.wins_sell / frame.trades_sell : 0.0;

#ifdef WR_RETURN_PERCENT
   wr_all  *= 100.0;
   wr_buy  *= 100.0;
   wr_sell *= 100.0;
#endif

   frame.overall_wr = wr_all;
   frame.buy_wr     = wr_buy;
   frame.sell_wr    = wr_sell;

   // --- Profit Factor ---
   if(gross_loss > 0.0)
      frame.profit_factor = gross_profit / gross_loss;
   else
      frame.profit_factor = (gross_profit > 0.0) ? 99.0 : 0.0;

   // --- Z-счёт (runs test) ---
   frame.zscore = 0.0;
   int W = (int)frame.wins;
   int L = (int)frame.losses;
   int N = ArraySize(res);
   if(W > 0 && L > 0 && N >= 2)
     {
      int runs = 1;
      for(int i = 1; i < N; i++)
         if(res[i] != res[i - 1])
            runs++;

      double n  = (double)(W + L);
      double mu = (2.0 * W * L) / n + 1.0;
      double denom = (2.0 * W * L * (2.0 * W * L - n)) / (n * n * (n - 1.0));
      double sigma = (denom > 0.0) ? MathSqrt(denom) : 0.0;
      if(sigma > 0.0)
         frame.zscore = (runs - mu) / sigma;
     }

   // --- Просадка баланса (% от пикового баланса) ---
   // Начальный баланс ~ текущий баланс - суммарный P/L за тест
   double initial = AccountInfoDouble(ACCOUNT_BALANCE) - frame.profit_sum;
   double peak_bal = initial + peak;
   if(peak_bal > 0.0 && max_dd > 0.0)
      frame.dd_balance_pct = max_dd / peak_bal * 100.0;
   frame.dd_equity_pct = frame.dd_balance_pct;   // оценка (без внутрисделочной кривой)

   string suffix = "";
#ifdef WR_RETURN_PERCENT
   suffix = "%";
#endif

   PrintFormat("TargetWinRate: %.0f закрытых сделок, прибыльных %.0f, winrate = %.2f%s, BUY=%.2f%s (%.0f/%.0f), SELL=%.2f%s (%.0f/%.0f), P/L = %.2f, PF = %.2f, Z = %.2f, DD = %.2f%%",
               frame.trades, frame.wins,
               wr_all, suffix,
               wr_buy, suffix, frame.wins_buy, frame.trades_buy,
               wr_sell, suffix, frame.wins_sell, frame.trades_sell,
               frame.profit_sum, frame.profit_factor, frame.zscore, frame.dd_balance_pct);
   return(wr_all);
  }
//+------------------------------------------------------------------+
//| OnTester: возвращает значение критерия для оптимизации          |
//| и добавляет кадр с метриками                                     |
//+------------------------------------------------------------------+
double OnTester()
  {
   TargetWinRateFrame frame;
   double wr = TargetWinRate_Collect(frame);

#ifdef WR_FRAME_REPORT
   // --- Кадр для отчёта/оптимизации ---
   TargetWinRateFrame frames[];
   ArrayResize(frames, 1);
   frames[0] = frame;

   long frame_id = 0;
   if(!FrameAdd("WinRate", frame_id, wr, frames))
      PrintFormat("TargetWinRate: ошибка FrameAdd (код %d)", GetLastError());
#endif

   return(wr);
  }
//+------------------------------------------------------------------+
//| OnTesterInit: создание CSV-отчёта до начала оптимизации        |
//+------------------------------------------------------------------+
int OnTesterInit()
  {
#ifdef WR_CSV_DUMP
   g_wr_csv_handle = INVALID_HANDLE;

   // Создаём файл заново (удаляем старый, если есть)
   FileDelete(WR_CSV_FILENAME, FILE_COMMON);
   ResetLastError();
   g_wr_csv_handle = FileOpen(WR_CSV_FILENAME, FILE_WRITE | FILE_READ | FILE_CSV | FILE_COMMON, ';');
   if(g_wr_csv_handle == INVALID_HANDLE)
      PrintFormat("TargetWinRate: не удалось создать CSV '%s' (код %d)", WR_CSV_FILENAME, GetLastError());
   else
     {
      // Заголовок
      FileWrite(g_wr_csv_handle, "pass", "winrate_total_%", "winrate_buy_%", "winrate_sell_%",
                "trades_total", "trades_buy", "trades_sell", "wins_buy", "wins_sell", "profit");
      PrintFormat("TargetWinRate: CSV-отчёт создан: Common\\Files\\%s", WR_CSV_FILENAME);
     }
#endif
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| OnTesterPass: чтение кадров, вывод в журнал и запись в CSV      |
//+------------------------------------------------------------------+
#ifdef WR_FRAME_REPORT
void OnTesterPass()
  {
   ulong  pass;
   string name;
   long   id;
   double value;
   TargetWinRateFrame frames[];

   while(FrameNext(pass, name, id, value, frames))
     {
      if(ArraySize(frames) < 1)
         continue;

      TargetWinRateFrame f = frames[0];
      PrintFormat("TargetWinRate[pass %I64u]: общий %.2f%% (%.0f сделок) | BUY %.2f%% (%.0f/%.0f) | SELL %.2f%% (%.0f/%.0f) | PF %.2f | Z %.2f | DD %.2f%% | P/L %.2f",
                  pass, f.overall_wr, f.trades,
                  f.buy_wr, f.wins_buy, f.trades_buy,
                  f.sell_wr, f.wins_sell, f.trades_sell,
                  f.profit_factor, f.zscore, f.dd_balance_pct, f.profit_sum);

#ifdef WR_CSV_DUMP
      if(g_wr_csv_handle != INVALID_HANDLE)
        {
         FileWrite(g_wr_csv_handle,
                   (string)pass,
                   DoubleToString(f.overall_wr, 2),
                   DoubleToString(f.buy_wr, 2),
                   DoubleToString(f.sell_wr, 2),
                   DoubleToString(f.trades, 0),
                   DoubleToString(f.trades_buy, 0),
                   DoubleToString(f.trades_sell, 0),
                   DoubleToString(f.wins_buy, 0),
                   DoubleToString(f.wins_sell, 0),
                   DoubleToString(f.profit_sum, 2));
        }
#endif
     }
  }
#endif // WR_FRAME_REPORT
//+------------------------------------------------------------------+
//| OnTesterDeinit: закрытие CSV-файла после оптимизации            |
//+------------------------------------------------------------------+
void OnTesterDeinit()
  {
#ifdef WR_CSV_DUMP
   if(g_wr_csv_handle != INVALID_HANDLE)
     {
      FileClose(g_wr_csv_handle);
      g_wr_csv_handle = INVALID_HANDLE;
     }
   PrintFormat("TargetWinRate: отчёт сохранён в Common\\Files\\%s", WR_CSV_FILENAME);
#endif
  }

#endif // TARGET_WINRATE_CRITERION_MQH
