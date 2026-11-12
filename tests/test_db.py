import unittest
from unittest.mock import Mock, patch

import mysql.connector
from fastapi import HTTPException

import db


class DatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_connection_and_cursor_close_when_insert_fails(self):
        cursor = Mock()
        cursor.execute.side_effect = mysql.connector.Error('sensitive internal detail')
        connection = Mock()
        connection.cursor.return_value = cursor
        with patch.object(db, '_connect', return_value=connection):
            with self.assertRaises(HTTPException) as result:
                await db.store_analysis_result('image-1', 'reports/one.pdf')
        self.assertEqual(result.exception.status_code, 500)
        self.assertNotIn('sensitive', result.exception.detail)
        cursor.close.assert_called_once()
        connection.close.assert_called_once()
        connection.commit.assert_not_called()
