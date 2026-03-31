import { useState, useEffect, useRef } from 'react';
import { generateSelectionSortSteps, type SortStep } from './SelectionSort';

const initialArray = [
    { value: 29, id: '1' },
    { value: 10, id: '2' },
    { value: 14, id: '3' },
    { value: 37, id: '4' },
    { value: 14, id: '5' },
    { value: 7, id: '6' }
];

function App() {
    const [arraySize, setArraySize] = useState(10);
    const [inputList, setInputList] = useState('');
    const [steps, setSteps] = useState<SortStep[]>([]);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(500);
    const [activeTab, setActiveTab] = useState<'explanation' | 'notes' | 'log'>('explanation');

    const playingRef = useRef(isPlaying);
    playingRef.current = isPlaying;

    const setupSteps = (items: { value: number; id: string }[]) => {
        const sortedSteps = generateSelectionSortSteps(items);
        setSteps(sortedSteps);
        setCurrentStepIndex(0);
        setIsPlaying(false);
    };

    useEffect(() => {
        setupSteps([...initialArray]);
    }, []);

    useEffect(() => {
        let timeoutId: number;
        if (isPlaying && currentStepIndex < steps.length - 1) {
            timeoutId = window.setTimeout(() => {
                setCurrentStepIndex(prev => prev + 1);
            }, speed);
        } else if (isPlaying && currentStepIndex >= steps.length - 1) {
            setIsPlaying(false);
        }
        return () => clearTimeout(timeoutId);
    }, [isPlaying, currentStepIndex, steps.length, speed]);

    const handleRandomize = () => {
        const newArray = Array.from({ length: arraySize }, (_, i) => ({
            value: Math.floor(Math.random() * 100) + 1,
            id: `rnd-${Date.now()}-${i}`
        }));
        setupSteps(newArray);
    };

    const handleCustomInput = () => {
        const vals = inputList.split(',').map(s => s.trim()).filter(s => s !== '');
        const newArray = vals.map((v, i) => ({
            value: parseInt(v) || 0,
            id: `cust-${Date.now()}-${i}`
        }));
        if (newArray.length > 0) {
            setupSteps(newArray);
        }
    };

    const currentStep = steps[currentStepIndex];

    const getBarState = (index: number) => {
        if (!currentStep) return 'default';
        if (currentStep.type === 'finalize' && index <= currentStep.i) return 'sorted';
        if (currentStep.type === 'complete') return 'sorted';
        if (index < currentStep.i) return 'sorted';

        if (currentStep.type === 'compare') {
            if (index === currentStep.j) return 'compare';
        }
        
        if (index === currentStep.minIndex) return 'min';
        if (index === currentStep.i) return 'i';
        
        return 'default';
    };

    const getBarHeight = (value: number) => {
        const maxVal = Math.max(...(currentStep?.array.map(a => a.value) || [100]), 100);
        return `${Math.max((value / maxVal) * 350, 20)}px`;
    };

    return (
        <div className="app-container">
            <header className="header">
                <h1>Selection Sort Sandbox</h1>
                <div className="legend">
                    <div className="legend-item"><div className="legend-color" style={{backgroundColor: 'var(--bar-sorted)'}}></div> Sorted</div>
                    <div className="legend-item"><div className="legend-color" style={{backgroundColor: 'var(--bar-i)'}}></div> Current 'i' (target)</div>
                    <div className="legend-item"><div className="legend-color" style={{backgroundColor: 'var(--bar-j)'}}></div> Scanning 'j'</div>
                    <div className="legend-item"><div className="legend-color" style={{backgroundColor: 'var(--bar-min)'}}></div> Min Candidate</div>
                    <div className="legend-item"><div className="legend-color" style={{backgroundColor: 'var(--bar-compare)'}}></div> Comparing</div>
                </div>
            </header>

            <main className="main-content">
                <section className="sandbox-area">
                    <div className="visualization">
                        {currentStep && currentStep.array.map((item, index) => (
                            <div key={item.id} className="array-bar-container">
                                <div 
                                    className="array-bar"
                                    data-state={getBarState(index)}
                                    style={{ height: getBarHeight(item.value) }}
                                >
                                    {item.value}
                                </div>
                                <div className="bar-label">{index}</div>
                            </div>
                        ))}
                    </div>

                    <div className="controls-panel">
                        <div className="control-group">
                            <button onClick={() => { setIsPlaying(false); setCurrentStepIndex(0); }}>⏮ Restart</button>
                            <button onClick={() => setCurrentStepIndex(p => Math.max(0, p - 1))} disabled={currentStepIndex === 0}>◀ Prev</button>
                            <button className="primary" onClick={() => setIsPlaying(!isPlaying)} disabled={currentStepIndex >= steps.length - 1}>
                                {isPlaying ? '⏸ Pause' : '▶ Play'}
                            </button>
                            <button onClick={() => setCurrentStepIndex(p => Math.min(steps.length - 1, p + 1))} disabled={currentStepIndex >= steps.length - 1}>Next ▶</button>
                            <button onClick={() => setCurrentStepIndex(steps.length - 1)}>⏭ End</button>
                        </div>
                        
                        <div className="control-group" style={{width: '100%', padding: '0 1rem'}}>
                            <input 
                                type="range" 
                                className="history-slider"
                                min={0} 
                                max={Math.max(0, steps.length - 1)} 
                                value={currentStepIndex}
                                onChange={(e) => { setIsPlaying(false); setCurrentStepIndex(parseInt(e.target.value)); }}
                            />
                        </div>

                        <div className="control-group">
                            <label>Speed (ms)</label>
                            <select value={speed} onChange={e => setSpeed(parseInt(e.target.value))}>
                                <option value={1000}>Slow (1000)</option>
                                <option value={500}>Normal (500)</option>
                                <option value={100}>Fast (100)</option>
                                <option value={10}>Instant (10)</option>
                            </select>
                        </div>

                        <div className="control-group">
                            <label>Size</label>
                            <input type="number" value={arraySize} onChange={e => setArraySize(parseInt(e.target.value))} style={{width: '60px'}}/>
                            <button onClick={handleRandomize}>Randomize</button>
                        </div>

                        <div className="control-group">
                            <input type="text" placeholder="e.g. 5, 2, 8, 5" value={inputList} onChange={e => setInputList(e.target.value)} />
                            <button onClick={handleCustomInput}>Set Data</button>
                        </div>
                    </div>
                </section>

                <aside className="sidebar">
                    <div className="tabs">
                        <button className={`tab ${activeTab === 'explanation' ? 'active' : ''}`} onClick={() => setActiveTab('explanation')}>Explanation</button>
                        <button className={`tab ${activeTab === 'notes' ? 'active' : ''}`} onClick={() => setActiveTab('notes')}>Notes</button>
                    </div>

                    <div className="tab-content">
                        {activeTab === 'explanation' && currentStep && (
                            <>
                                <div className="explanation-panel">
                                    <strong>Step {currentStepIndex} / {steps.length - 1}</strong>
                                    <p>{currentStep.description}</p>
                                </div>
                                <div className="metrics-box">
                                    <div className="metric">
                                        <div>Comparisons</div>
                                        <div className="value">{currentStep.comparisons}</div>
                                    </div>
                                    <div className="metric">
                                        <div>Swaps</div>
                                        <div className="value">{currentStep.swaps}</div>
                                    </div>
                                    <div className="metric">
                                        <div>Total Expected O(n²)</div>
                                        <div className="value" style={{fontSize: '1rem', marginTop: '1rem'}}>
                                            ~{Math.floor((currentStep.array.length * (currentStep.array.length - 1)) / 2)} compares
                                        </div>
                                    </div>
                                </div>
                            </>
                        )}
                        
                        {activeTab === 'notes' && (
                            <div className="notes-panel">
                                <h2>Selection Sort</h2>
                                <p><strong>Time Complexity:</strong> O(n²) all cases.</p>
                                <p><strong>Space Complexity:</strong> O(1) in-place.</p>
                                <h3>Algorithm</h3>
                                <ul>
                                    <li>Divide array into sorted prefix and unsorted suffix.</li>
                                    <li>Find the minimum element in the unsorted suffix.</li>
                                    <li>Swap it with the first element of the unsorted suffix.</li>
                                    <li>Expand the sorted prefix by 1.</li>
                                </ul>
                                <h3>Properties</h3>
                                <p><strong>Not Stable:</strong> Selection sort can swap equal elements past each other (e.g. if you have [4a, 4b, 2], 4a gets swapped with 2, making the array [2, 4b, 4a]).</p>
                                <p><strong>Low Swaps:</strong> It places exactly one item in its final position per pass, making it useful when write operations are expensive.</p>
                            </div>
                        )}
                    </div>
                </aside>
            </main>
        </div>
    );
}

export default App;