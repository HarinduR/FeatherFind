class SPARQLQueryBuilder:
    def build_query(self, features):
        prefixes = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            PREFIX ff: <http://www.semanticweb.org/daham/ontologies/2025/0/feather-finder#>
        """

        conditions = ["?bird a ff:Bird"]
        
        # Helper function to format values
        def format_value(value):
            return value.replace(" ", "_").capitalize()

        if features['color']['primary']:
            color = format_value(features['color']['primary'])
            conditions.append(f"ff:hasPrimaryColor ff:{color}")
            
        if features['color']['secondary']:
            color = format_value(features['color']['secondary'])
            conditions.append(f"ff:hasSecondaryColor ff:{color}")
            
        if features['habitat']:
            habitat = format_value(features['habitat'])
            conditions.append(f"ff:livesIn ff:{habitat}")
            
        if features['region']:
            region = format_value(features['region'])
            conditions.append(f"ff:locatedIn ff:{region}")
            
        if features.get('diet'): 
            diet = format_value(features['diet'])
            conditions.append(f"ff:eats ff:{diet}")
            
        if features['size']:
            size = format_value(features['size'])
            conditions.append("ff:hasSize ff:{}".format(size))  

        if features['beak']['size']:
            beakSize = format_value(features['beak']['size'])
            conditions.append(f"ff:hasBeak ff:{beakSize}")
        
        if features['beak']['color']:
            beakColor = format_value(features['beak']['color'])
            conditions.append(f"ff:hasBeakColor ff:{beakColor}")
        
        if features['eyes']['size']:
            eyesSize = format_value(features['eyes']['size'])
            conditions.append(f"ff:hasEyes ff:{eyesSize}")
        
        if features['eyes']['color']:
            eyesColor = format_value(features['eyes']['color'])
            conditions.append(f"ff:hasEyesColor ff:{eyesColor}")
        
        if features['legs']['size']:
            legsSize = format_value(features['legs']['size'])
            conditions.append(f"ff:hasLegs ff:{legsSize}")
        
        if features['legs']['color']:
            legsColor = format_value(features['legs']['color'])
            conditions.append(f"ff:haslegsColor ff:{legsColor}")

        where_clause = " ;\n        ".join(conditions) + " ."

        return f"""
        {prefixes}
        SELECT ?bird
        WHERE {{
            {where_clause}
        }}
        LIMIT 5
        """