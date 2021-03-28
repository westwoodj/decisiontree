def findDecision(obj): #obj[0]: X_coord, obj[1]: Y_coord
   # {"feature": "Y_coord", "instances": 750000, "metric_value": 0.9179, "depth": 1}
   if obj[1]>210.95714167321046:
      # {"feature": "X_coord", "instances": 591796, "metric_value": 0.7423, "depth": 2}
      if obj[0]>0:
         return 'pos'
      elif obj[0]<=0:
         return 'pos'
      else:
         return 'pos'
   elif obj[1]<=210.95714167321046:
      # {"feature": "X_coord", "instances": 158204, "metric_value": 0.7393, "depth": 2}
      if obj[0]>0:
         return 'neg'
      elif obj[0]<=0:
         return 'neg'
      else:
         return 'neg'
   else:
      return 'neg'
