"""
Automated Test Runner for Document Assistant
Runs all test questions and generates a comprehensive report
"""

import sys
import os
import json
import io
import pandas as pd
from datetime import datetime
from pathlib import Path
import time

# Ensure UTF-8 output so emojis/logs don't crash on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_assistant import ConversationalRAG

def run_test_suite():
    """Run complete test suite and generate report"""
    
    print("="*80)
    print("DOCUMENT ASSISTANT - AUTOMATED TEST SUITE")
    print("="*80)
    print(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    # Initialize the RAG system
    print("\n📚 Initializing Document Assistant...")
    start_time = time.time()
    
    try:
        rag = ConversationalRAG("Gen AI/Source")
        init_time = time.time() - start_time
        print(f"✅ System initialized in {init_time:.2f} seconds")
        print(f"📄 Loaded {len(rag.all_chunks)} document chunks")
        
        # Count documents by type
        doc_types = {}
        for chunk in rag.all_chunks:
            doc_types[chunk.source_type] = doc_types.get(chunk.source_type, 0) + 1
        
        print(f"📊 Document distribution: {doc_types}")
        
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    # Test questions
    test_questions = [
        {
            "id": 1,
            "question": "What is the order quantity handled by Ranjit?",
            "category": "Structured Query",
            "expected_type": "numeric"
        },
        {
            "id": 2,
            "question": "What is the percentage of orders that haven't dispatched?",
            "category": "Statistical Query",
            "expected_type": "percentage"
        },
        {
            "id": 3,
            "question": "List of products under the recoil kits orders?",
            "category": "List Query",
            "expected_type": "list"
        },
        {
            "id": 4,
            "question": "How much GST was charged for my insurance and when does my third party insurance expire?",
            "category": "Document Extraction",
            "expected_type": "compound"
        },
        {
            "id": 5,
            "question": "What are the roles we are currently hiring for?",
            "category": "HR Query",
            "expected_type": "text"
        },
        {
            "id": 6,
            "question": "For the product Glock 17 what is the planned WO release date?",
            "category": "Product Query",
            "expected_type": "date"
        },
        {
            "id": 7,
            "question": "Ok try Glock - 17",
            "category": "Alternative Format",
            "expected_type": "text"
        },
        {
            "id": 8,
            "question": "What is our criteria to hire a data scientist?",
            "category": "Policy Query",
            "expected_type": "text"
        },
        {
            "id": 9,
            "question": "What are the benifits of log book and how can i set it up?",
            "category": "Technical Documentation",
            "expected_type": "text"
        },
        {
            "id": 10,
            "question": "How do i create a zone?",
            "category": "Process Query",
            "expected_type": "text"
        }
    ]
    
    print(f"\n🧪 Running {len(test_questions)} test questions...")
    print("-"*80)
    
    # Store results
    test_results = []
    successful_tests = 0
    
    # Run each test
    for test in test_questions:
        print(f"\n📝 Test {test['id']}: {test['question']}")
        print(f"   Category: {test['category']}")
        
        start_time = time.time()
        
        try:
            # Get response
            response = rag.generate_response(test['question'])
            query_time = time.time() - start_time
            
            # Evaluate response
            has_answer = bool(response['answer'] and response['answer'] != "I couldn't find specific information about your query in the documents.")
            has_sources = bool(response['sources'])
            has_data = bool(response.get('data'))
            
            # Determine if test passed
            test_passed = has_answer
            
            if test_passed:
                successful_tests += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            # Store result
            test_results.append({
                'Test ID': test['id'],
                'Question': test['question'],
                'Category': test['category'],
                'Status': status,
                'Has Answer': has_answer,
                'Has Sources': has_sources,
                'Has Data': has_data,
                'Response Time (s)': round(query_time, 3),
                'Answer Length': len(response['answer']),
                'Answer Preview': response['answer'][:200] + "..." if len(response['answer']) > 200 else response['answer']
            })
            
            # Print result
            print(f"   Status: {status}")
            print(f"   Response Time: {query_time:.3f}s")
            print(f"   Answer: {response['answer'][:100]}...")
            if response['sources']:
                print(f"   Sources: {', '.join(set(response['sources']))}")
            if has_data:
                print(f"   Structured Data: Available")
                
        except Exception as e:
            test_results.append({
                'Test ID': test['id'],
                'Question': test['question'],
                'Category': test['category'],
                'Status': '❌ ERROR',
                'Has Answer': False,
                'Has Sources': False,
                'Has Data': False,
                'Response Time (s)': 0,
                'Answer Length': 0,
                'Answer Preview': f"Error: {str(e)}"
            })
            print(f"   Status: ❌ ERROR - {str(e)}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    total_tests = len(test_questions)
    pass_rate = (successful_tests / total_tests) * 100
    
    print(f"\n📊 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {successful_tests}")
    print(f"   Failed: {total_tests - successful_tests}")
    print(f"   Pass Rate: {pass_rate:.1f}%")
    
    # Calculate statistics
    df_results = pd.DataFrame(test_results)
    avg_response_time = df_results['Response Time (s)'].mean()
    
    print(f"\n⏱️ Performance Metrics:")
    print(f"   Average Response Time: {avg_response_time:.3f}s")
    print(f"   Max Response Time: {df_results['Response Time (s)'].max():.3f}s")
    print(f"   Min Response Time: {df_results['Response Time (s)'].min():.3f}s")
    
    # Category analysis
    print(f"\n📈 Category Analysis:")
    category_stats = df_results.groupby('Category').agg({
        'Status': lambda x: sum('PASSED' in str(s) for s in x),
        'Response Time (s)': 'mean'
    }).rename(columns={'Status': 'Passed', 'Response Time (s)': 'Avg Time (s)'})
    
    for category, stats in category_stats.iterrows():
        print(f"   {category}: {int(stats['Passed'])} passed, {stats['Avg Time (s)']:.3f}s avg")
    
    # Save detailed report
    report_path = "test_results.csv"
    df_results.to_csv(report_path, index=False)
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    # Generate JSON report for programmatic access
    json_report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total_tests,
            'passed': successful_tests,
            'failed': total_tests - successful_tests,
            'pass_rate': pass_rate,
            'avg_response_time': avg_response_time
        },
        'test_results': test_results
    }
    
    json_path = "test_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"📋 JSON report saved to: {json_path}")
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Document Assistant Test Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #1f77b4; }}
            .summary {{ background: #f0f2f6; padding: 15px; border-radius: 5px; }}
            .passed {{ color: green; font-weight: bold; }}
            .failed {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #1f77b4; color: white; }}
            tr:nth-child(even) {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Document Assistant Test Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Total Tests: {total_tests}</p>
            <p class="passed">Passed: {successful_tests}</p>
            <p class="failed">Failed: {total_tests - successful_tests}</p>
            <p>Pass Rate: {pass_rate:.1f}%</p>
            <p>Average Response Time: {avg_response_time:.3f}s</p>
        </div>
        
        <h2>Detailed Results</h2>
        {df_results.to_html(index=False)}
    </body>
    </html>
    """
    
    html_path = "test_report.html"
    with open(html_path, 'w') as f:
        f.write(html_report)
    print(f"🌐 HTML report saved to: {html_path}")
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    return test_results

def main():
    """Main entry point"""
    try:
        results = run_test_suite()
        
        # Print final message
        print("\n✅ All tests completed. Check the generated reports for detailed results.")
        print("\nGenerated files:")
        print("  - test_results.csv   (Detailed CSV report)")
        print("  - test_results.json  (JSON format for programmatic access)")
        print("  - test_report.html   (HTML report for viewing in browser)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test suite interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
