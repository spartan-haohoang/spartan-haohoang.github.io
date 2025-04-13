---
layout: ../../layouts/post.astro
title: Understanding the Foundations of Repository-Level AI Software Engineering with RepoGraph
description: Introducing RepoGraph, a graph-based module that maps out the structure of an entire codebase
dateFormatted: February 20, 2025
---

# I. Overview

## Introduction

Imagine you’re tasked with fixing a bug in a massive codebase, like the one powering a popular open-source library. You can’t just focus on a single function or file—you need to understand how dozens of files, modules, and dependencies interact. Now, imagine handing this task to an AI. While today’s large language models (LLMs) excel at writing small, self-contained snippets of code, they often stumble when faced with the complexity of real-world software repositories. This is where the research behind *REPOGRAPH* comes in—a groundbreaking approach to help AI tackle modern software engineering challenges at the repository level.

If you’re an AI practitioner, researcher, or enthusiast, this blog series will unpack why repository-level understanding is the next frontier for AI-driven software engineering and how REPOGRAPH is paving the way. Whether you’re building AI tools, contributing to open-source projects, or simply curious about the future of coding, this research offers insights that could shape how we interact with codebases in the AI era.

## Research Context

Large language models (LLMs) have transformed how we approach coding. From generating quick scripts to fixing bugs in isolated functions, tools like Code-Llama and StarCoder have shown impressive results. But real-world software engineering isn’t just about writing or fixing small pieces of code—it’s about managing entire repositories. These repositories are complex ecosystems of interdependent files, modules, and libraries, and tasks like adding features, resolving GitHub issues, or ensuring changes don’t break existing functionality require a deep, holistic understanding of the codebase.

The problem? Most AI tools today are designed for function-level or file-level tasks. They struggle to “see” the bigger picture, like how a change in one file might ripple through the entire repository. Existing approaches, like retrieval-augmented generation (RAG) or agent-based frameworks, try to address this by retrieving relevant files or letting AI “explore” the codebase. However, these methods often fall short—they either focus on semantic similarity without understanding true dependencies or get stuck in local optima, missing the global structure of the repository.

This gap is what *REPOGRAPH* aims to bridge. The researchers behind this paper, from institutions like the University of Illinois Urbana-Champaign and Tencent AI Seattle Lab, set out to create a tool that helps AI understand and navigate codebases at the repository level. Their goal? To empower AI to tackle real-world software engineering tasks, like those evaluated in benchmarks like SWE-Bench, with greater accuracy and efficiency.

## Key Contributions

- **REPOGRAPH: A Plug-In Module for Repository-Level Understanding**
    
    The researchers introduce REPOGRAPH, a graph-based module that maps out the structure of an entire codebase. Unlike previous methods that treat repositories as flat documents or focus on file-level analysis, REPOGRAPH operates at the line level, capturing fine-grained dependencies between code definitions and references.
    
- **Graph-Based Representation**
    
    REPOGRAPH represents the repository as a graph, where each node is a line of code, and edges connect related lines based on dependencies. This structure allows AI to “see” the intricate relationships across files, making it easier to navigate complex codebases.
    
- **Sub-Graph Retrieval for Contextual Insights**
    
    Using sub-graph retrieval algorithms, REPOGRAPH extracts focused “ego-graphs” centered around specific keywords or issues. These sub-graphs provide AI with targeted context, helping it make informed decisions for tasks like bug fixing or feature addition.
    
- **Significant Performance Boosts**
    
    When integrated with existing AI software engineering frameworks, REPOGRAPH delivers an average 32.8% improvement in success rates on the SWE-Bench benchmark. It also shows strong transferability to other repo-level tasks, as demonstrated on CrossCodeEval.
    
- **Extensibility and Flexibility**
    
    REPOGRAPH is designed as a plug-in module, meaning it can enhance both procedural and agent-based AI frameworks. Its versatility makes it a promising tool for a wide range of software engineering applications.
    

## Takeaways for Readers

By diving into this research, you’ll gain:

- **A New Perspective on AI Coding Challenges**: Understand why repository-level tasks are the next big hurdle for AI and why current tools fall short.

- **Practical Insights for AI Tool Development**: Learn how graph-based approaches like REPOGRAPH can improve AI’s ability to navigate and modify complex codebases, inspiring ideas for your own projects or tools.

- **A Glimpse into the Future of Software Engineering**: See how AI could evolve to handle real-world software tasks, from resolving GitHub issues to contributing to open-source projects.

- **Inspiration for Research or Collaboration**: Discover opportunities to build on this work, whether you’re a researcher exploring AI-driven software engineering or a practitioner looking to integrate REPOGRAPH into your workflow.

## What’s Next

In this first part, we’ve explored the “why” behind REPOGRAPH and the challenges it aims to solve. But how does it actually work? In the next part, we’ll dive into the technical details and methodology behind this research, unpacking the graph-based approach, sub-graph retrieval algorithms, and how REPOGRAPH integrates with existing frameworks. Stay tuned for a deeper look at the “how” behind this game-changing tool!

# II. Diving into the Methodology of Repository-Level AI Software Engineering with REPOGRAPH

In the first part of this series, we explored why repository-level understanding is a game-changer for AI-driven software engineering and how *REPOGRAPH* aims to tackle this challenge. While large language models (LLMs) excel at function-level tasks, they struggle with the complexity of real-world codebases—think interdependent files, intricate dependencies, and the need to make changes without breaking existing functionality. REPOGRAPH, a graph-based plug-in module, promises to help AI navigate these complexities by mapping out the structure of entire repositories.

Now, it’s time for a technical deep dive. In this part, we’ll unpack how REPOGRAPH works under the hood, from its construction process to its integration with existing AI frameworks. Whether you’re a researcher, developer, or AI enthusiast, this breakdown will help you understand the “how” behind this innovative approach. Let’s get started!

## Overview of the Methodology

At its core, REPOGRAPH is like a GPS for AI navigating a codebase. Instead of treating a repository as a flat collection of files, it builds a structured graph that maps out the relationships between lines of code. This graph helps AI trace dependencies, understand execution flow, and pinpoint the root cause of issues—key for tasks like fixing bugs or adding features in complex repositories.

The methodology can be broken down into three main steps:

1. **Code Line Parsing**: REPOGRAPH starts by scanning the repository, identifying relevant code files, and parsing them into a detailed structure using tools like tree-sitter.

2. **Dependency Filtering**: It then filters out irrelevant relationships (e.g., calls to built-in Python functions) to focus on project-specific dependencies.

3. **Graph Construction**: Finally, it builds a graph where nodes represent lines of code, and edges capture dependencies, creating a map of the repository’s structure.

Once built, REPOGRAPH can be used to retrieve focused sub-graphs (called “ego-graphs”) around specific keywords or issues. These sub-graphs provide AI with targeted context, making it easier to solve repository-level tasks. REPOGRAPH is designed to plug into both procedural and agent-based AI frameworks, enhancing their ability to navigate and modify codebases.

## Technical Details

Let’s dive deeper into the technical aspects of REPOGRAPH, breaking down its construction, representation, and integration.

### Key Components and Steps

1. **Code Line Parsing (Step 1)**
    - REPOGRAPH starts by traversing the repository to identify code files (e.g., `.py` files) while ignoring irrelevant ones (e.g., `.git` or `requirements.txt`).
    - It uses *tree-sitter*, a parsing tool that generates an Abstract Syntax Tree (AST) for each file. Think of the AST as a blueprint of the code, highlighting key elements like functions, classes, and variables.
    - The AST not only identifies definitions (e.g., `class Model`) but also tracks where these elements are referenced (e.g., `self._validate_input_units()`).
    - REPOGRAPH focuses on lines involving function calls and dependencies, discarding less relevant details like individual variables.
    - **Example**: For a Python file, REPOGRAPH might identify `class Model` as a definition and `self.prepare_inputs()` as a reference, capturing their relationship.
    
    *Suggestion for Visual*: A diagram showing a sample code snippet, its AST, and how REPOGRAPH extracts definitions and references would clarify this step.
    
2. **Project-Dependent Relation Filtering (Step 2)**
    - After parsing, REPOGRAPH has a list of code lines with relationships (e.g., function calls). However, not all relationships are useful—calls to built-in functions like `len()` or third-party libraries can distract from project-specific dependencies.
    - REPOGRAPH filters out two types of irrelevant relations:
        - *Global relations*: Calls to Python’s standard or built-in libraries (e.g., `len`, `list`). These are excluded using a pre-built list of standard methods.
        - *Local relations*: Calls to third-party libraries, identified by parsing import statements.
    - **Example**: In the line `inputs = len(input)`, `len` is excluded because it’s a built-in function, leaving only project-specific relations.
    - This filtering ensures REPOGRAPH focuses on the repository’s unique structure, making it more efficient for AI tasks.
    
    *Suggestion for Visual*: A flowchart showing the filtering process, with examples of global and local relations being removed, would help illustrate this step.
    
3. **Graph Construction (Step 3)**
    - REPOGRAPH builds a graph `G = {V, E}`, where:
        - *V (Nodes)*: Each node represents a line of code, with attributes like `line_number`, `file_name`, and `directory`. Nodes are classified as:
            - *Definition nodes (“def”)*: Lines where functions or classes are defined (e.g., `class Model`).
            - *Reference nodes (“ref”)*: Lines where definitions are used (e.g., `self.prepare_inputs()`).
        - *E (Edges)*: Edges capture relationships between nodes, with two types:
            - *E_contain*: Connects a definition node to its internal components (e.g., a class to its methods).
            - *E_invoke*: Connects a definition node to its references (e.g., a function to where it’s called).
    - **Example**: For `class Model`, the definition node might have `E_contain` edges to its methods and `E_invoke` edges to lines where `Model` is referenced.
    - This graph structure allows AI to trace dependencies across files, providing a holistic view of the repository.
    
    *Suggestion for Visual*: A graph diagram showing nodes (def/ref) and edges (contain/invoke) for a sample repository would make this concept more tangible.
    

### Novel Techniques and Innovations

- **Line-Level Granularity**: Unlike previous methods that analyze repositories at the file level, REPOGRAPH operates at the line level. This fine-grained approach captures detailed dependencies, making it easier for AI to navigate complex codebases.
- **Ego-Graph Retrieval**: REPOGRAPH uses k-hop ego-graphs to retrieve focused sub-graphs around specific search terms (e.g., `separability_matrix`). These sub-graphs provide AI with targeted context, reducing noise and improving decision-making.
    - **Intuitive Explanation**: Think of ego-graphs as “zooming in” on a specific part of the repository, like focusing on a neighborhood in a city map.
- **Integration with Existing Frameworks**: REPOGRAPH is designed as a plug-in module, enhancing both procedural and agent-based AI frameworks.
*Suggestion for Visual*: A side-by-side comparison of how REPOGRAPH integrates with procedural vs. agent frameworks, showing sample prompts or actions, would clarify this innovation.
    - *Procedural frameworks*: REPOGRAPH’s sub-graphs are appended to prompts at key stages (e.g., localization, edition), helping LLMs make informed decisions.
    - *Agent frameworks*: REPOGRAPH adds a new action (`search_repograph`) to the agent’s toolkit, allowing it to query the graph for dependencies.

### Mathematical Concepts (Simplified)

- **Graph Representation**: REPOGRAPH’s graph is formally defined as `G = {V, E}`, where:
    - `V` is the set of nodes (lines of code).
    - `E` is the set of edges (relationships).
    - Edges are categorized as `E_contain` (containment) and `E_invoke` (invocation).
- **Ego-Graph Retrieval**: For a search term (e.g., `separability_matrix`), REPOGRAPH retrieves a k-hop ego-graph, which includes:
    - The central node (search term).
    - All nodes within k hops (levels of connection).
    - **Intuitive Explanation**: If k=2, the ego-graph includes the search term, its direct dependencies (1 hop), and their dependencies (2 hops).
- **Flattening for Integration**: The retrieved ego-graph is “flattened” (converted into a list of node attributes) for use in prompts or agent actions.
    
    *Suggestion for Visual*: A diagram showing a sample ego-graph (central node + k-hop neighbors) and its flattened output would help explain this concept.
    

## Challenges and Trade-offs

While REPOGRAPH is innovative, it comes with challenges:

- **Scalability**: Parsing large repositories with tree-sitter and building detailed graphs can be computationally expensive. The authors don’t explicitly address scalability, but filtering irrelevant relations helps mitigate this issue.

- **Noise in Parsing**: Tree-sitter may miss some dependencies or misinterpret complex code structures. The authors focus on functions and classes to reduce noise, but this might exclude other relevant details (e.g., variables).

- **Filtering Trade-offs**: Excluding built-in and third-party relations reduces noise but risks missing important context. For example, a third-party library might be critical for understanding a bug.

- **Integration Complexity**: Adding REPOGRAPH to existing frameworks requires modifying prompts or agent actions, which could be challenging for practitioners without deep technical expertise.

The authors address some challenges (e.g., filtering noise) but leave others open, such as scalability and handling edge cases in parsing. These limitations highlight areas for future research or optimization.

## Practical Implications

REPOGRAPH’s methodology has exciting implications for AI-driven software engineering:

- **Real-World Applications**:

- *Bug Fixing*: REPOGRAPH can help AI trace dependencies to find the root cause of issues, like fixing a bug in a library like Astropy.

- *Feature Addition*: By mapping out the repository, AI can identify where to add new features without breaking

# III. Evaluating the Impact of Repository-Level AI Software Engineering with REPOGRAPH: Results and Insights

In the previous parts of this series, we explored why repository-level understanding is critical for AI-driven software engineering and how *REPOGRAPH* tackles this challenge with its graph-based approach. By mapping out the structure of entire codebases, REPOGRAPH helps AI navigate complex dependencies, making it easier to fix bugs, add features, and resolve real-world issues. We also dove into the technical details, from parsing code with tree-sitter to retrieving focused ego-graphs for context.

Now, it’s time to see how REPOGRAPH performs in action. In this final part, we’ll unpack the experiments, results, and insights from the research, evaluating whether REPOGRAPH lives up to its promise. Whether you’re an AI researcher, developer, or enthusiast, these findings will help you understand the impact of this approach and its potential for real-world applications. Let’s dive into the results!

## Overview of Experiments

The researchers tested REPOGRAPH by integrating it as a plug-in module into existing AI software engineering frameworks, evaluating its performance on a challenging benchmark. Here’s a simple breakdown of the experimental setup:

- **Dataset**:
    - The experiments used *SWE-bench Lite*, a benchmark that tests AI’s ability to solve real-world software issues.
    - Each task requires submitting a patch (code changes) to fix a described issue, ensuring all test scripts pass.
    - Think of it as AI being asked to fix bugs or add features in open-source projects, like resolving a GitHub issue.
- **Evaluation Metrics**:
    - *Accuracy*:
        - *Resolve rate*: Percentage of issues successfully resolved (i.e., patches pass all tests).
        - *Patch application rate*: Percentage of patches that can be applied to the repository without errors.
    - *Cost Efficiency*:
        - *Average cost*: Dollar cost of running the AI (based on LLM inference costs).
        - *Average tokens*: Number of input/output tokens used when querying LLMs.
    - These metrics balance effectiveness (does it work?) with efficiency (how much does it cost?).
- **Baselines and Comparisons**:
    - REPOGRAPH was plugged into two types of AI frameworks:
        - *Procedural frameworks*: Traditional methods like RAG (retrieval-augmented generation) and Agentless (a state-of-the-art open-source approach).
        - *Agent frameworks*: SWE-agent and AutoCodeRover, which let AI dynamically explore and act on the codebase.
    - The same LLM versions (GPT-4 and GPT-4o) were used for fair comparisons.
    - REPOGRAPH’s performance was compared to these baselines to measure improvements.

All experiments were run in a Docker environment for reproducibility, with procedural frameworks taking 2-3 hours and agent frameworks up to 10 hours per run.

## Key Results

The results show that REPOGRAPH significantly boosts performance across all frameworks, with some trade-offs in cost. Below is a summary of the main findings, including quantitative improvements and qualitative insights.

- **Quantitative Results (Accuracy and Cost)**:
    - REPOGRAPH improved resolve rates (issues fixed) and patch application rates (patches applied) for all frameworks.
    - Best performance: Agentless + REPOGRAPH achieved a 29.67% resolve rate, setting a new state-of-the-art for open-source methods on SWE-bench Lite.
    - Detailed results are shown in the table below (data from the paper’s Table 2).
    
    *Suggestion for Visual*: A bar chart comparing resolve rates and patch application rates for baselines vs. +REPOGRAPH would make these improvements stand out.
    
    | Framework | LLM | Resolve Rate (%) | # Samples Resolved | Patch Apply Rate (%) | Avg. Cost ($) | Avg. Tokens |
    | --- | --- | --- | --- | --- | --- | --- |
    | **Procedural Frameworks** |  |  |  |  |  |  |
    | RAG | GPT-4 | 2.67 | 8 | 29.33 | $0.13 | 11,736 |
    | +REPOGRAPH | GPT-4 | 5.33 (+2.66) | 16 (+8) | 47.67 (+18.34) | $0.17 | 15,439 |
    | Agentless | GPT-4o | 27.33 | 82 | 97.33 | $0.34 | 42,376 |
    | +REPOGRAPH | GPT-4o | 29.67 (+2.34) | 89 (+7) | 98.00 (+0.67) | $0.39 | 47,323 |
    | **Agent Frameworks** |  |  |  |  |  |  |
    | AutoCodeRover | GPT-4 | 19.00 | 57 | 83.00 | $0.45 | 38,663 |
    | +REPOGRAPH | GPT-4 | 21.33 (+2.33) | 64 (+7) | 86.67 (+3.67) | $0.58 | 45,112 |
    | SWE-agent | GPT-4o | 18.33 | 55 | 87.00 | $2.51 | 245,008 |
    | +REPOGRAPH | GPT-4o | 20.33 (+2.00) | 61 (+6) | 90.33 (+3.33) | $2.69 | 262,512 |
- **Qualitative Insights**:
    - REPOGRAPH consistently improved performance across all frameworks, with larger gains in procedural frameworks (e.g., +2.66% for RAG, +2.34% for Agentless).
    - Procedural frameworks benefited more because their structured, deterministic nature makes it easier to integrate REPOGRAPH’s context.
    - Agent frameworks saw smaller gains, likely due to their dynamic, exploratory nature, which can lead to redundant or costly actions.
    - Cost increases were modest for procedural frameworks but higher for agent frameworks, especially SWE-agent (+$0.18), due to frequent use of the `search_repograph` action.
    - Importantly, performance gains were not just due to increased token usage—REPOGRAPH’s context improved decision-making efficiency.

## Discussion and Analysis

The authors analyzed the results to understand what worked, what didn’t, and why. Here’s a breakdown of their insights, with relatable analogies and examples.

- **What Worked Well and Why**:
    - *Consistent Gains Across Frameworks*: REPOGRAPH’s graph-based context helped AI make more informed decisions, like a GPS guiding you through a city instead of relying on vague directions.
        - Example: For RAG, REPOGRAPH doubled the resolve rate (2.67% to 5.33%) by providing precise dependency information, reducing guesswork.
    - *Stronger Impact on Procedural Frameworks*: These frameworks, with their step-by-step workflows, leveraged REPOGRAPH’s context more effectively.
        - Analogy: It’s like following a recipe (procedural) vs. improvising in the kitchen (agent)—the recipe benefits more from precise ingredient lists (REPOGRAPH).
    - *State-of-the-Art Performance*: Agentless + REPOGRAPH’s 29.67% resolve rate shows REPOGRAPH’s potential to push the boundaries of open-source AI tools.
- **What Didn’t Work and Potential Reasons**:
    - *Smaller Gains in Agent Frameworks*: Agents’ dynamic nature led to less efficient use of REPOGRAPH.
        - Example: SWE-agent might call `search_repograph` repeatedly, like over-checking a map, increasing costs without proportional benefits.
    - *Cost Increases in Agent Frameworks*: The exploratory nature of agents led to higher token usage and costs.
        - Analogy: It’s like taking multiple detours on a road trip (agent) vs. following a direct route (procedural)—detours cost more fuel (tokens).
    - *Baseline Performance Matters*: Frameworks with stronger baselines (e.g., Agentless) saw more absolute gains, while weaker baselines (e.g., RAG) saw larger relative improvements.
- **Surprising or Unexpected Findings**:
    - *Cost Efficiency*: Despite higher costs in agent frameworks, REPOGRAPH’s gains weren’t just due to more tokens—it improved decision quality, not just quantity.
        - Example: It’s like upgrading your car’s navigation system (REPOGRAPH) vs. just driving more miles (tokens)—better navigation gets you there faster.
    - *Procedural vs. Agent Gap*: The authors expected agent frameworks to benefit more due to their flexibility, but procedural frameworks outperformed expectations.

## Limitations and Future Work

While REPOGRAPH shows promise, it has limitations that could affect practical applications:

- **Scalability**: Building and querying graphs for large repositories can be computationally expensive, especially for agent frameworks with long runtimes (up to 10 hours).

  - *Impact*: This could limit REPOGRAPH’s use in massive projects without optimization.

- **Cost in Agent Frameworks**: Higher costs (e.g., +$0.18 for SWE-agent) make REPOGRAPH less cost-effective for exploratory agents.

  - *Impact*: Users need to balance performance gains with budget constraints, especially in industry settings.
