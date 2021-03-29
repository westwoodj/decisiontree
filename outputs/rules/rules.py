def findDecision(obj): #obj[0]: X_coord, obj[1]: Y_coord, obj[2]: rad_diff, obj[3]: rad_squared
   # {"feature": "rad_diff", "instances": 750000, "metric_value": 0.9335, "depth": 1}
   if obj[2]>61161.387936590254:
      # {"feature": "X_coord", "instances": 605618, "metric_value": 0.7102, "depth": 2}
      if obj[0]>183.92038988443989:
         # {"feature": "Y_coord", "instances": 467610, "metric_value": 0.8077, "depth": 3}
         if obj[1]>176.3493904825611:
            # {"feature": "rad_squared", "instances": 359446, "metric_value": 0.9044, "depth": 4}
            if obj[3]<=111111.1111111111:
               return 'neg'
            else:
               return 'neg'
         elif obj[1]<=176.3493904825611:
            # {"feature": "rad_squared", "instances": 108164, "metric_value": 0.0631, "depth": 4}
            if obj[3]<=111111.1111111111:
               return 'neg'
            else:
               return 'neg'
         else:
            return 'neg'
      elif obj[0]<=183.92038988443989:
         # {"feature": "Y_coord", "instances": 138008, "metric_value": 0.0994, "depth": 3}
         if obj[1]>211.20243023829556:
            # {"feature": "rad_squared", "instances": 108768, "metric_value": 0.1204, "depth": 4}
            if obj[3]<=111111.1111111111:
               return 'neg'
            else:
               return 'neg'
         elif obj[1]<=211.20243023829556:
            return 'neg'
         else:
            return 'neg'
      else:
         return 'neg'
   elif obj[2]<=61161.387936590254:
      return 'pos'
   else:
      return 'pos'
