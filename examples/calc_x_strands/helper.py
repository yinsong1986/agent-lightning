def extract_numbers_and_percentages(text):
    """
    Extracts integers, floats, percentages, and fractions from a string.
    Supports numbers with comma separators like 1,000.
    Returns the last extracted numerical value as a float.
    """
    import re
    extracted_values = []
    processed_spans = []  # Track the character spans we've already processed

    # Pattern for fractions like 1/3, -2/5, etc.
    fraction_pattern = r"[-+]?\d+/\d+(?![^\W\d])"
    
    # Find all fraction matches with their positions
    fraction_matches = [(m.group(), m.start(), m.end()) 
                        for m in re.finditer(fraction_pattern, text)]
    
    for match, start, end in fraction_matches:
        try:
            numerator, denominator = match.split('/')
            value = float(numerator) / float(denominator)
            extracted_values.append(value)
            # Mark this text span as processed
            processed_spans.append((start, end))
        except (ValueError, ZeroDivisionError):
            # Handle invalid fractions or division by zero
            pass

    # Regex for numbers (integers or floats) and percentages with comma separators
    number_pattern = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:%)?(?![^\W\d])"
    
    # Find all number matches with their positions
    number_matches = [(m.group(), m.start(), m.end()) 
                      for m in re.finditer(number_pattern, text)]

    for match, start, end in number_matches:
        # Check if this number is part of an already processed fraction
        if any(start >= ps and end <= pe for ps, pe in processed_spans):
            continue  # Skip this number as it's part of a fraction
            
        if match.endswith('%'):
            try:
                # Remove '%' and convert to float (after removing commas), then divide by 100
                value = float(match[:-1].replace(',', '')) / 100
                extracted_values.append(value)
            except ValueError:
                pass
        else:
            try:
                # Remove commas and convert to float (handles both integers and floats)
                value = float(match.replace(',', ''))
                extracted_values.append(value)
            except ValueError:
                pass
    
    # Return the last number as the answer
    if extracted_values:
        return extracted_values[-1]
    return None