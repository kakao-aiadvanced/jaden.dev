from openai import OpenAI
client = OpenAI()

content = """
These are some examples that natural language to SQL conversion.
1. "Find all employees whose salary is over 50000.": SELECT * FROM employees WHERE salary > 50000;
2. "Find all products that is out of stock.": SELECT * FROM products WHERE stock = 0;
3. "Find all student's name whose math score is over 90.": SELECT name FROM students WHERE math_score > 90;
4. "Find all orders which ordered in 30 days before now.": SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY);
5. "Count customers count in certain cities.": SELECT city, COUNT(*) FROM customers GROUP BY city;

My Request: "Find the average salary of employees in the marketing department."
give me converted SQL Query:
"""

print(content)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": content
        }
    ]
)

answer = completion.choices[0].message

print(answer.content)
print()
print("role: ", answer.role)