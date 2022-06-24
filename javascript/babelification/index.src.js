import React from 'react';
import { createRoot } from 'react-dom/client';
import Button from 'react-bootstrap/Button';
console.log("Hello world!");


const container = document.getElementById('root');
const root = createRoot(container); // createRoot(container!) if you use TypeScript

root.render(<div>
  <Button variant="primary">Primary</Button>{' '}
</div>);