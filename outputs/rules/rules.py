def findDecision(obj): #obj[0]: X_coord, obj[1]: Y_coord
   # {"feature": "Y_coord", "instances": 750000, "metric_value": 0.8133, "depth": 1}
   if obj[1]<=788.2936449934562:
      # {"feature": "X_coord", "instances": 591648, "metric_value": 0.9028, "depth": 2}
      if obj[0]<=788.1703980617405:
         return 'neg'
      elif obj[0]>788.1703980617405:
         return 'neg'
      else:
         return 'neg'
   elif obj[1]>788.2936449934562:
      return 'neg'
   else:
      return 'neg'
