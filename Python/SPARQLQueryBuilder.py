class SPARQLQueryBuilder:
    def build_query(self, features):
        
        query = """
            PREFIX ex: <http://example.org/birds#>
            SELECT ?bird WHERE {
            ?bird a ex:Bird .
        """

        if features['color']['primary']:
            color_value = features['color']['primary'].capitalize()
            query += f'?bird ex:hasPrimaryColor ex:{color_value} .\n'
        if features['color']['secondary']:
            color_value = features['color']['secondary'].capitalize()
            query += f'?bird ex:hasSecondaryColor ex:{color_value} .\n'
        if features['habitat']:
            habitat_value = features['habitat'].capitalize()
            query += f'?bird ex:livesIn ex:{habitat_value} .\n'
        if features['region']:
            region_value = features['region'].replace(" ", "").capitalize()
            query += f'?bird ex:locatedIn ex:{region_value} .\n'
        query += "} LIMIT 5"

        return query