from __future__ import annotations
import re
import pandas as pd

PRIORITY_CATEGORIES=[
 ('delivery_problem',['delivery','deliver','package','parcel','shipment','shipped','courier','driver']),
 ('order_status',['order','ordered','preorder','arrive','arrival']),
 ('return_or_refund',['refund','return','reimburse']),
 ('prime_membership',['prime','membership','subscribe','subscription']),
 ('account_or_security',['account','password','login','sign in','hacked','unauthor','fraud','security']),
 ('payment_or_pricing',['charged','charge','payment','price','billing','card','cost']),
 ('product_availability',['stock','available','availability','item','product','listing']),
 ('technical_or_content',['app','website','web site','error','bug','technical','stream','video','kindle']),
 ('customer_service',['customer service','customer care','contact','call','representative','manager','support team']),
 ('other_or_unclear',[]),
]

def assign_intents(messages:pd.Series)->pd.Series:
 texts=messages.fillna('').astype(str).str.lower()
 assigned=pd.Series(False,index=messages.index); labels=pd.Series('other_or_unclear',index=messages.index)
 for name,keywords in PRIORITY_CATEGORIES:
  if not keywords: continue
  mask=texts.str.contains('|'.join(re.escape(k) for k in keywords),regex=True) & ~assigned
  labels.loc[mask]=name; assigned|=mask
 return labels
