//+------------------------------------------------------------------+
//|                                            RNNFilter_News.mqh    |
//|   Фильтр новостей (экономический календарь MQL5) для советника   |
//|   RNNFilter.mq5.                                                 |
//|   Активируется только при Inp_NewsFilterEnable == true           |
//|   (вызов из OnTick).                                             |
//+------------------------------------------------------------------+
#ifndef RNNFILTER_NEWS_MQH
#define RNNFILTER_NEWS_MQH
//+------------------------------------------------------------------+
//| Проверка: есть ли важные события календаря в окне [from, to]     |
//| Параметры:                                                       |
//|   impact_threshold - минимальная важность события (1-3)          |
//|   window_before    - окно ДО события, минуты                     |
//|   window_after     - окно ПОСЛЕ события, минуты                  |
//| Возвращает true, если торговлю нужно заблокировать               |
//+------------------------------------------------------------------+
bool IsNewsBlocked(int impact_threshold, int window_before, int window_after)
  {
   MqlCalendarValue values[];
   datetime from = TimeCurrent() - window_before * 60;
   datetime to   = TimeCurrent() + window_after * 60;

   int total = CalendarValueHistory(values, from, to);
   for(int i = 0; i < total; i++)
     {
      MqlCalendarEvent ev;
      if(!CalendarEventById(values[i].event_id, ev))
         continue;

      // Если важность события >= порога — блокируем
      if((int)ev.importance >= impact_threshold)
        {
         PrintFormat("Новость заблокирована: %s (важность: %d)",
                     ev.name, (int)ev.importance);
         return(true);
        }
     }

   return(false);
  }
//+------------------------------------------------------------------+
#endif // RNNFILTER_NEWS_MQH
