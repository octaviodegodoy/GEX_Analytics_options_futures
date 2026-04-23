//+------------------------------------------------------------------+
//| GEX_Walls.mq5 — GEX Levels overlay from Python CSV              |
//| Reads MQL5/Files/GEX_<Symbol>.csv and draws horizontal lines    |
//| for walls, gamma flip, pins, support & resistance zones.        |
//+------------------------------------------------------------------+
#property copyright "GEX Analytics"
#property version   "2.00"
#property indicator_chart_window
#property indicator_plots 0

//--- Inputs
input string InpSymbol       = "BOVA11";   // Symbol name in CSV filename
input int    InpUpdateSec    = 10;         // Reload interval (seconds)
input bool   InpShowPins     = true;       // Show pin candidate lines
input bool   InpShowZones    = true;       // Show support/resistance zones
input bool   InpShowWeekly   = true;       // Show per-week walls
input bool   InpShowComment  = true;       // Show dashboard comment

//--- Level storage
double g_spot, g_call_wall, g_put_wall, g_gamma_flip, g_regime;

// Entry lines (nearest support / resistance zone)
double g_entry_buy, g_entry_sell;
double g_win_entry_buy, g_win_entry_sell;

// Trade signal
double g_signal;
string g_signal_name;
int    g_signal_strength;
string g_signal_regime;

// Current WIN symbol from CSV
string g_win_symbol;

// Per-week walls (up to 2 weeks)
double g_wk_call_wall[2], g_wk_put_wall[2], g_wk_flip[2];
string g_wk_expiry[2];
int    g_wk_count;

// Pins (up to 5)
double g_pins[5];
int    g_pin_count;

// Resistance / Support (up to 3 each)
double g_resist[3], g_support[3];
int    g_resist_count, g_support_count;

// WIN equivalents
double g_win_spot, g_win_call_wall, g_win_put_wall, g_win_gamma_flip;
double g_win_pins[5], g_win_resist[3], g_win_support[3];
double g_win_wk_call_wall[2], g_win_wk_put_wall[2], g_win_wk_flip[2];
bool   g_has_win;

//--- Auto-detect: draw at WIN prices when chart is a futures symbol
bool   g_use_win;

//--- Contract mismatch: chart symbol vs CSV win_symbol
bool   g_contract_mismatch;

//--- Object prefix for cleanup
#define OBJ_PREFIX "GEX_"

//+------------------------------------------------------------------+
//| Pick the right price for the chart: WIN or BOVA11               |
//+------------------------------------------------------------------+
double Pick(double bova, double win)
{
   if(g_use_win && win > 0) return win;
   return bova;
}

//+------------------------------------------------------------------+
int OnInit()
{
   // Auto-detect: if chart symbol contains WIN or IND, use WIN prices
   string chart_sym = _Symbol;
   StringToUpper(chart_sym);
   g_use_win = (StringFind(chart_sym, "WIN") >= 0 || StringFind(chart_sym, "IND") >= 0);
   g_contract_mismatch = false;

   EventSetTimer(InpUpdateSec);
   ReadCSV();
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   DeleteAllObjects();
   Comment("");
   EventKillTimer();
}

//+------------------------------------------------------------------+
void OnTimer()
{
   ReadCSV();
}

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
{
   return(rates_total);
}

//+------------------------------------------------------------------+
void ReadCSV()
{
   string filename = "GEX_" + InpSymbol + ".csv";
   int handle = FileOpen(filename, FILE_READ | FILE_CSV | FILE_ANSI, ',');

   if(handle == INVALID_HANDLE)
   {
      Print("Cannot open: ", filename, " Error: ", GetLastError());
      if(InpShowComment)
         Comment("GEX: Waiting for data...\nRun Python script to generate CSV.");
      return;
   }

   // Reset
   g_spot = 0; g_call_wall = 0; g_put_wall = 0; g_gamma_flip = 0; g_regime = 0;
   g_entry_buy = 0; g_entry_sell = 0;
   g_win_entry_buy = 0; g_win_entry_sell = 0;
   g_signal = 0; g_signal_name = ""; g_signal_strength = 0; g_signal_regime = "";
   g_win_symbol = "";
   g_wk_count = 0; g_pin_count = 0; g_resist_count = 0; g_support_count = 0;
   g_has_win = false;
   g_win_spot = 0; g_win_call_wall = 0; g_win_put_wall = 0; g_win_gamma_flip = 0;
   ArrayInitialize(g_pins, 0);
   ArrayInitialize(g_resist, 0);
   ArrayInitialize(g_support, 0);
   ArrayInitialize(g_wk_call_wall, 0);
   ArrayInitialize(g_wk_put_wall, 0);
   ArrayInitialize(g_wk_flip, 0);
   ArrayInitialize(g_win_pins, 0);
   ArrayInitialize(g_win_resist, 0);
   ArrayInitialize(g_win_support, 0);
   ArrayInitialize(g_win_wk_call_wall, 0);
   ArrayInitialize(g_win_wk_put_wall, 0);
   ArrayInitialize(g_win_wk_flip, 0);

   // Skip header: key,value,win,expiry
   FileReadString(handle); FileReadString(handle);
   FileReadString(handle); FileReadString(handle);

   int wk_cw_idx = 0, wk_pw_idx = 0, wk_fl_idx = 0;

   while(!FileIsEnding(handle))
   {
      string key    = FileReadString(handle);
      string sval   = FileReadString(handle);
      string swin   = FileReadString(handle);
      string expiry = FileReadString(handle);

      double val = StringToDouble(sval);
      double win = StringToDouble(swin);
      if(swin != "" && win != 0) g_has_win = true;

      // --- Core levels ---
      if(key == "spot")        { g_spot = val;        g_win_spot = win; }
      if(key == "call_wall")   { g_call_wall = val;   g_win_call_wall = win; }
      if(key == "put_wall")    { g_put_wall = val;    g_win_put_wall = win; }
      if(key == "gamma_flip")  { g_gamma_flip = val;  g_win_gamma_flip = win; }
      if(key == "regime")        g_regime = val;

      // --- Entry lines ---
      if(key == "entry_buy")    { g_entry_buy = val;    g_win_entry_buy = win; }
      if(key == "entry_sell")   { g_entry_sell = val;   g_win_entry_sell = win; }

      // --- Trade signal ---
      if(key == "signal")            g_signal = val;
      if(key == "signal_name")       g_signal_name = sval;
      if(key == "signal_strength")   g_signal_strength = (int)val;
      if(key == "signal_regime")     g_signal_regime = sval;

      // --- Current WIN symbol ---
      if(key == "win_symbol")        g_win_symbol = sval;

      // --- Weekly walls ---
      if(StringFind(key, "_call_wall") >= 0 && wk_cw_idx < 2)
      {
         g_wk_call_wall[wk_cw_idx] = val;
         g_win_wk_call_wall[wk_cw_idx] = win;
         if(wk_cw_idx < 2) g_wk_expiry[wk_cw_idx] = expiry;
         wk_cw_idx++;
         if(wk_cw_idx > g_wk_count) g_wk_count = wk_cw_idx;
      }
      if(StringFind(key, "_put_wall") >= 0 && wk_pw_idx < 2)
      {
         g_wk_put_wall[wk_pw_idx] = val;
         g_win_wk_put_wall[wk_pw_idx] = win;
         wk_pw_idx++;
      }
      if(StringFind(key, "_flip") >= 0 && StringFind(key, "gamma_flip") < 0 && wk_fl_idx < 2)
      {
         g_wk_flip[wk_fl_idx] = val;
         g_win_wk_flip[wk_fl_idx] = win;
         wk_fl_idx++;
      }

      // --- Pin candidates ---
      if(StringFind(key, "pin_") == 0 && g_pin_count < 5)
      {
         g_pins[g_pin_count] = val;
         g_win_pins[g_pin_count] = win;
         g_pin_count++;
      }

      // --- Resistance ---
      if(StringFind(key, "resist_") == 0 && g_resist_count < 3)
      {
         g_resist[g_resist_count] = val;
         g_win_resist[g_resist_count] = win;
         g_resist_count++;
      }

      // --- Support ---
      if(StringFind(key, "support_") == 0 && g_support_count < 3)
      {
         g_support[g_support_count] = val;
         g_win_support[g_support_count] = win;
         g_support_count++;
      }
   }

   FileClose(handle);

   // Re-evaluate g_use_win: if CSV provides a win_symbol, check
   // whether this chart matches it or any WIN/IND chart
   if(g_win_symbol != "")
   {
      string chart_sym_up = _Symbol;
      StringToUpper(chart_sym_up);
      g_use_win = (StringFind(chart_sym_up, "WIN") >= 0 || StringFind(chart_sym_up, "IND") >= 0);

      // Detect contract mismatch: chart symbol vs current futures contract
      string win_sym_up = g_win_symbol;
      StringToUpper(win_sym_up);
      if(g_use_win && StringFind(chart_sym_up, "$") < 0)  // skip continuous symbols like WIN$N
      {
         // Chart is on a specific contract — check if it matches
         g_contract_mismatch = (chart_sym_up != win_sym_up);
         if(g_contract_mismatch)
            Print("[GEX] Contract mismatch: chart=", _Symbol, " but current contract=", g_win_symbol,
                  ". Consider switching chart to ", g_win_symbol);
      }
      else
         g_contract_mismatch = false;
   }
   else
      g_contract_mismatch = false;

   DeleteAllObjects();
   DrawLevels();
}

//+------------------------------------------------------------------+
string WinLabel(double bova, double win)
{
   string sym_tag = (g_win_symbol != "") ? g_win_symbol : "WIN";
   if(!g_has_win || win == 0)
      return DoubleToString(bova, 2);
   return DoubleToString(bova, 2) + " (" + sym_tag + " " + DoubleToString(win, 0) + ")";
}

//+------------------------------------------------------------------+
// Label placement: collect prices, then stagger overlapping labels
double g_label_prices[];
string g_label_names[];
string g_label_texts[];
color  g_label_colors[];
int    g_label_count;

void ResetLabels() { g_label_count = 0; ArrayResize(g_label_prices, 0); ArrayResize(g_label_names, 0); ArrayResize(g_label_texts, 0); ArrayResize(g_label_colors, 0); }

void QueueLabel(string name, double price, string text, color clr)
{
   int n = g_label_count;
   g_label_count = n + 1;
   ArrayResize(g_label_prices, n + 1);
   ArrayResize(g_label_names,  n + 1);
   ArrayResize(g_label_texts,  n + 1);
   ArrayResize(g_label_colors, n + 1);
   g_label_prices[n] = price;
   g_label_names[n]  = name;
   g_label_texts[n]  = text;
   g_label_colors[n] = clr;
}

void FlushLabels()
{
   if(g_label_count == 0) return;

   // Sort labels by price (ascending) using simple insertion sort
   for(int i = 1; i < g_label_count; i++)
   {
      double  tp = g_label_prices[i];
      string  tn = g_label_names[i];
      string  tt = g_label_texts[i];
      color   tc = g_label_colors[i];
      int j = i - 1;
      while(j >= 0 && g_label_prices[j] > tp)
      {
         g_label_prices[j+1] = g_label_prices[j];
         g_label_names[j+1]  = g_label_names[j];
         g_label_texts[j+1]  = g_label_texts[j];
         g_label_colors[j+1] = g_label_colors[j];
         j--;
      }
      g_label_prices[j+1] = tp;
      g_label_names[j+1]  = tn;
      g_label_texts[j+1]  = tt;
      g_label_colors[j+1] = tc;
   }

   // Assign horizontal column offsets: when prices are too close, shift right
   int cols[];
   ArrayResize(cols, g_label_count);
   cols[0] = 0;

   // Minimum vertical gap in price points to avoid overlap (~font height)
   double min_gap = SymbolInfoDouble(_Symbol, SYMBOL_POINT) * 200;
   if(min_gap <= 0) min_gap = 5.0;  // fallback for index

   for(int i = 1; i < g_label_count; i++)
   {
      if(MathAbs(g_label_prices[i] - g_label_prices[i-1]) < min_gap)
         cols[i] = cols[i-1] + 1;
      else
         cols[i] = 0;
   }

   int bar_shift = PeriodSeconds() * 3;
   int col_width = PeriodSeconds() * 18;  // horizontal spacing per column

   for(int i = 0; i < g_label_count; i++)
   {
      string lbl_name = OBJ_PREFIX + "L_" + g_label_names[i];
      datetime anchor_time = TimeCurrent() + bar_shift + cols[i] * col_width;
      ObjectCreate(0, lbl_name, OBJ_TEXT, 0, anchor_time, g_label_prices[i]);
      ObjectSetString(0, lbl_name, OBJPROP_TEXT, g_label_texts[i]);
      ObjectSetInteger(0, lbl_name, OBJPROP_COLOR, g_label_colors[i]);
      ObjectSetInteger(0, lbl_name, OBJPROP_FONTSIZE, 8);
      ObjectSetString(0, lbl_name, OBJPROP_FONT, "Arial Bold");
      ObjectSetDouble(0, lbl_name, OBJPROP_ANGLE, 0);
      ObjectSetInteger(0, lbl_name, OBJPROP_ANCHOR, ANCHOR_LEFT_LOWER);
      ObjectSetInteger(0, lbl_name, OBJPROP_SELECTABLE, false);
      ObjectSetInteger(0, lbl_name, OBJPROP_BACK, false);
   }
}

void MakeHLine(string name, double price, color clr, int width,
               ENUM_LINE_STYLE style, string label_text)
{
   if(price <= 0) return;
   string obj_name = OBJ_PREFIX + name;
   ObjectCreate(0, obj_name, OBJ_HLINE, 0, 0, price);
   ObjectSetInteger(0, obj_name, OBJPROP_COLOR, clr);
   ObjectSetInteger(0, obj_name, OBJPROP_WIDTH, width);
   ObjectSetInteger(0, obj_name, OBJPROP_STYLE, style);
   ObjectSetInteger(0, obj_name, OBJPROP_BACK, true);
   ObjectSetInteger(0, obj_name, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, obj_name, OBJPROP_TEXT, label_text);
   ObjectSetString(0, obj_name, OBJPROP_TOOLTIP, label_text);

   // Queue label for staggered placement
   QueueLabel(name, price, label_text, clr);
}

//+------------------------------------------------------------------+
void DrawLevels()
{
   ResetLabels();

   // --- Combined walls ---
   MakeHLine("CallWall", Pick(g_call_wall, g_win_call_wall), clrDodgerBlue, 2, STYLE_SOLID,
             "Call Wall: " + WinLabel(g_call_wall, g_win_call_wall));

   MakeHLine("PutWall", Pick(g_put_wall, g_win_put_wall), clrOrangeRed, 2, STYLE_SOLID,
             "Put Wall: " + WinLabel(g_put_wall, g_win_put_wall));

   MakeHLine("GammaFlip", Pick(g_gamma_flip, g_win_gamma_flip), clrGold, 2, STYLE_DASH,
             "Gamma Flip: " + WinLabel(g_gamma_flip, g_win_gamma_flip));

   // --- Entry lines (nearest S/R zone to spot) ---
   MakeHLine("EntryBuy", Pick(g_entry_buy, g_win_entry_buy), clrLime, 2, STYLE_DASHDOTDOT,
             "ENTRY BUY (S/R): " + WinLabel(g_entry_buy, g_win_entry_buy));

   MakeHLine("EntrySell", Pick(g_entry_sell, g_win_entry_sell), clrRed, 2, STYLE_DASHDOTDOT,
             "ENTRY SELL (S/R): " + WinLabel(g_entry_sell, g_win_entry_sell));

   // --- Per-week walls ---
   if(InpShowWeekly)
   {
      for(int i = 0; i < g_wk_count; i++)
      {
         string si = IntegerToString(i + 1);
         string exp = g_wk_expiry[i];

         MakeHLine("Wk" + si + "_CW", Pick(g_wk_call_wall[i], g_win_wk_call_wall[i]),
                   clrCornflowerBlue, 1, STYLE_DOT,
                   "Wk" + si + " Call Wall (" + exp + "): " +
                   WinLabel(g_wk_call_wall[i], g_win_wk_call_wall[i]));

         MakeHLine("Wk" + si + "_PW", Pick(g_wk_put_wall[i], g_win_wk_put_wall[i]),
                   clrCoral, 1, STYLE_DOT,
                   "Wk" + si + " Put Wall (" + exp + "): " +
                   WinLabel(g_wk_put_wall[i], g_win_wk_put_wall[i]));

         MakeHLine("Wk" + si + "_Flip", Pick(g_wk_flip[i], g_win_wk_flip[i]),
                   clrKhaki, 1, STYLE_DASHDOT,
                   "Wk" + si + " Flip (" + exp + "): " +
                   WinLabel(g_wk_flip[i], g_win_wk_flip[i]));
      }
   }

   // --- Pin candidates ---
   if(InpShowPins)
   {
      for(int i = 0; i < g_pin_count; i++)
      {
         string si = IntegerToString(i + 1);
         MakeHLine("Pin" + si, Pick(g_pins[i], g_win_pins[i]),
                   clrMediumOrchid, 1, STYLE_DOT,
                   "Pin " + si + ": " + WinLabel(g_pins[i], g_win_pins[i]));
      }
   }

   // --- Resistance zones ---
   if(InpShowZones)
   {
      for(int i = 0; i < g_resist_count; i++)
      {
         string si = IntegerToString(i + 1);
         MakeHLine("Resist" + si, Pick(g_resist[i], g_win_resist[i]),
                   clrDeepSkyBlue, 1, STYLE_DASHDOT,
                   "Resist " + si + ": " + WinLabel(g_resist[i], g_win_resist[i]));
      }

      for(int i = 0; i < g_support_count; i++)
      {
         string si = IntegerToString(i + 1);
         MakeHLine("Support" + si, Pick(g_support[i], g_win_support[i]),
                   clrSandyBrown, 1, STYLE_DASHDOT,
                   "Support " + si + ": " + WinLabel(g_support[i], g_win_support[i]));
      }
   }

   // Place all queued labels with staggered positions
   FlushLabels();

   // --- Dashboard comment ---
   if(InpShowComment)
   {
      string regime_str = (g_regime > 0) ? "POSITIVE (Low Vol)"
                        : (g_regime < 0) ? "NEGATIVE (High Vol)"
                        : "TRANSITION";

      string sym_tag = (g_win_symbol != "") ? g_win_symbol : "WIN";

      string txt = "=== GEX DASHBOARD — " + InpSymbol;
      if(g_win_symbol != "") txt += " (" + sym_tag + ")";
      txt += " ===\n";

      // Contract mismatch warning
      if(g_contract_mismatch)
         txt += "!! CHART MISMATCH: " + _Symbol + " != " + g_win_symbol
              + " - switch chart!\n";

      txt += "Regime:      " + regime_str + "\n"
         + "Gamma Flip:  " + WinLabel(g_gamma_flip, g_win_gamma_flip) + "\n"
         + "Call Wall:   " + WinLabel(g_call_wall, g_win_call_wall) + "  (Resistance)\n"
         + "Put Wall:    " + WinLabel(g_put_wall, g_win_put_wall) + "  (Support)\n"
         + "--- Entry Levels (Nearest S/R Zone) ---\n"
         + "Entry SELL:  " + WinLabel(g_entry_sell, g_win_entry_sell) + "\n"
         + "Entry BUY:   " + WinLabel(g_entry_buy, g_win_entry_buy) + "\n";

      if(g_signal_name != "")
      {
         txt += "--- Signal ---\n"
              + "Signal:   " + g_signal_name + " [" + IntegerToString(g_signal_strength) + "/3]\n"
              + "Regime:   " + g_signal_regime + "\n";
      }

      if(g_pin_count > 0)
      {
         txt += "--- Pin Candidates ---\n";
         for(int i = 0; i < g_pin_count; i++)
            txt += "  " + IntegerToString(i+1) + ". " + WinLabel(g_pins[i], g_win_pins[i]) + "\n";
      }

      if(g_resist_count > 0)
      {
         txt += "--- Resistance Zones ---\n";
         for(int i = 0; i < g_resist_count; i++)
            txt += "  " + IntegerToString(i+1) + ". " + WinLabel(g_resist[i], g_win_resist[i]) + "\n";
      }

      if(g_support_count > 0)
      {
         txt += "--- Support Zones ---\n";
         for(int i = 0; i < g_support_count; i++)
            txt += "  " + IntegerToString(i+1) + ". " + WinLabel(g_support[i], g_win_support[i]) + "\n";
      }

      txt += "==========================";
      Comment(txt);
   }

   ChartRedraw(0);
}

//+------------------------------------------------------------------+
void DeleteAllObjects()
{
   int total = ObjectsTotal(0, 0, -1);
   for(int i = total - 1; i >= 0; i--)
   {
      string name = ObjectName(0, i, 0, -1);
      if(StringFind(name, OBJ_PREFIX) == 0)
         ObjectDelete(0, name);
   }
}
//+------------------------------------------------------------------+
