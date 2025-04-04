import React, { useState } from 'react';
import './style.css';

const Square = ({ value, onClick }) => (
  <button className="square" onClick={onClick}>
    {value}
  </button>
);

const Board = ({ squares, onClick }) => (
  <div className="board">
    {squares.map((square, index) => (
      <Square key={index} value={square} onClick={() => onClick(index)} />
    ))}
  </div>
);

const Game = () => {
  const [history, setHistory] = useState([Array(9).fill(null)]);
  const [stepNumber, setStepNumber] = useState(0);
  const [gameState, setGameState] = useState('');

  const handleClick = (i) => {
    if (gameState === "O Ganhou!" || gameState === "X Ganhou!" || gameState === "Empate") {
        return;
      }
    if (!history[stepNumber][i]) {
      const current = history[stepNumber].slice();
      current[i] = 'X';
      const newHistory = history.slice(0, stepNumber + 1).concat([current]);
      setHistory(newHistory);
      setStepNumber(stepNumber + 1);
      
      const emptyIndices = [];
      current.forEach((square, index) => {
        if (!square) {
          emptyIndices.push(index);
        }
      });
      const randomIndex = emptyIndices[Math.floor(Math.random() * emptyIndices.length)];
      current[randomIndex] = 'O';
      const updatedHistory = newHistory.slice();
      updatedHistory.push(current);
      setHistory(updatedHistory);
      setStepNumber(stepNumber + 2);

      const squaresData = current.map(value => value === null ? 'b' : value.toLowerCase());
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ squares: squaresData })
      })
      .then(response => response.json())
      .then(data => {
        console.log('Received prediction:', data);
        setGameState(data.game_state);
      })
      .catch(error => {
        console.error('Error:', error);
      });
    }
  };

  const current = history[stepNumber];

  return (
    <div className="game">
      <div className="game-board">
        <Board squares={current} onClick={handleClick} />
      </div>
      <div className="game-info">
        <div>Game State: {gameState}</div>
      </div>
    </div>
  );
};

export default Game;
