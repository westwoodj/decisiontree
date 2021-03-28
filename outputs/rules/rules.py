def findDecision(obj): #obj[0]: X_coord, obj[1]: Y_coord
   # {"feature": "X_coord", "instances": 750000, "metric_value": 0.9335, "depth": 1}
   if obj[0]>210.84405368575352:
      # {"feature": "Y_coord", "instances": 591852, "metric_value": 0.986, "depth": 2}
      if obj[1]>210.90762271815464:
         return 'pos'
      elif obj[1]<=210.90762271815464:
         return 'neg'
      else:
         return 'neg'
   elif obj[0]<=210.84405368575352:
      # {"feature": "Y_coord", "instances": 158148, "metric_value": 0.27, "depth": 2}
      if obj[1]>211.14267549863615:
         return 'neg'
      elif obj[1]<=211.14267549863615:
         return 'neg'
      else:
         return 'neg'
   else:
      return 'neg'
