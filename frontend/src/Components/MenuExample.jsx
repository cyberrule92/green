import React from 'react';
import { Menu } from 'grommet';

export const MenuExample = ({ items, ...rest }) => {
  return <Menu label="Menu" items={items} width="medium" {...rest} />;
};